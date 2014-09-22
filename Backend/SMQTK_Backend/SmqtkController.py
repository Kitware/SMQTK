"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import atexit
import logging
import json
import multiprocessing
import os.path as osp
import tempfile
import types
import uuid

from .AsyncSearchWorker import AsyncSearchWorker
from .RefinableSearchState import RefinableSearchState
from .SharedAttribute import register_mdb_loc, SharedAttribute
from .utils import DatabaseInfo
from .utils import SafeConfigCommentParser
from .utils.jsmin import jsmin
from .VCDController import VCDController
from .ECDController import ECDController

from .HWIndexerInterface import (
    HWGlobalInitialize,
    HWSessionInitialize,
)


class SmqtkController (object):
    """
    Queue and control processes execution for the SMQTK system.

    Structure that controls execution of video and event content descriptors
    as well as event content fusion into an end result event content description
    that is query-able by a user via a search engine.

    For each search session initialized, there will always be a current IQR
    search state, but not an archive search state. Archive search states are
    only initialized when an archive search is requested (get calls will return
    KeyErrors until this is so).
    """

    @classmethod
    def generate_config(cls, config=None):
        """
        Create a default configuration object for this controller.

        :param config: Existing config object to extend. This may be none. In
            that case we create a new config object
        :type config: SafeConfigCommentParser or None
        :return: A config object now containing parameters relevant to this
            controller.
        :rtype: SafeConfigCommentParser

        """
        if config is None:
            config = SafeConfigCommentParser()

        sect = 'smqtk_controller'
        config.add_section(sect,
                           "General configuration parameters for the SMQTK "
                           "controller.")
        config.set(sect, 'work_directory',
                   osp.join(tempfile.gettempdir(), 'SmqtkWork'),
                   'The directory where generated, intermediate data will '
                   'be kept.')
        config.set(sect, 'data_directory',
                   osp.join(osp.abspath(osp.dirname(__file__)), 'data'),
                   'Root directory of video data storage.')

        config.set(sect, 'default_classifier_config_file',
                   osp.join(osp.abspath(osp.dirname(__file__)),
                            'data/ecd/classifier_config.json'),
                   'The path to the configuration file defining classifier '
                   'usage for different events. If a relative path is given, '
                   'it will be interpreted as relative to the configured data '
                   'directory.')

        config.set(sect, 'mongo_host', 'localhost',
                   'Host where the MongoDB mongod instance is running')
        config.set(sect, 'mongo_port', '27017',
                   'Port where the MongoDB mongod instance is running')
        config.set(sect, 'mongo_database', 'smqtk',
                   'The database in the configured MongoDB mongod instance to '
                   'utilize.')

        # Update config with sub-component configurations
        config = VCDController.generate_config(config)
        config = ECDController.generate_config(config)

        return config

    def __init__(self, config):
        """
        Initialize the SMQTK controller object

        :param config: Configuration object for this controller
        :type config: ConfigParser.ConfigParser

        """
        self._log = logging.getLogger('.'.join((self.__module__,
                                                self.__class__.__name__)))

        ### Helpers
        abspath = lambda p: osp.abspath(osp.expanduser(p))
        cget = lambda k: config.get(sect, k)
        cgetint = lambda k: config.getint(sect, k)

        ### Configuration extraction
        sect = 'smqtk_controller'
        self._work_dir = abspath(cget('work_directory'))
        self._data_dir = abspath(cget('data_directory'))
        self._mdb_info = DatabaseInfo(cget('mongo_host'),
                                      cgetint('mongo_port'),
                                      cget('mongo_database'))

        classifier_config_path = \
            osp.join(self._data_dir, cget('default_classifier_config_file'))
        # load JSON after stripping it of comments using jsmin
        self._default_classifier_config = json.loads(
            jsmin(open(classifier_config_path).read())
        )

        # Global SharedAttribute initialization
        # -> store location set to be the same location we have been configured
        #    to use.
        register_mdb_loc(self._mdb_info.host, self._mdb_info.port)
        SharedAttribute.DB_NAME = self._mdb_info.name

        self.__is_shutdown = False
        atexit.register(self.shutdown)

        ### Internal members

        # VCD and ECD controller sub-processes
        self.vcd_controller = VCDController(config)
        self.ecd_controller = ECDController(config, self.vcd_controller,
                                            self._mdb_info)

        # Starting controllers
        self.vcd_controller.start()
        self.ecd_controller.start()

        ###
        # IQR related things
        #

        # Defining fusion result model ID for top-down declaration (allows
        # immediate return of result location)
        self._result_mid = "FUSION"

        # Mapping of a session UUID (uuid.UUID objects) to state information
        # about searches for that session.

        # Maps a search UUID to that search's current IQR state.
        self._iqr_search_sessions_lock = multiprocessing.RLock()
        #: :type: dict of (uuid.UUID, RefinableSearchState)
        self._iqr_search_sessions = {}

        # Maps a search UUID to that search's current archive state
        self._archive_search_map_lock = multiprocessing.RLock()
        #: :type: dict of ((uuid.UUID, uuid.UUID), RefinableSearchState)
        self._archive_search_map = {}

        # Maps a search state to its associated search process (if there is one)
        #: :type: dict of (RefinableSearchState, AsyncSearchWorker)
        self._state_to_worker = {}

        HWGlobalInitialize()



    @property
    def data_directory(self):
        """
        :return: The path to the configured data directory.
        :rtype: str

        """
        return self._data_dir

    @property
    def work_directory(self):
        """
        :return: The path to the configured working directory.
        :rtype: str

        """
        return self._work_dir

    def shutdown(self):
        """
        Cleanly shutdown the Controller instance, transitively shutting down
        submodules.
        """
        if not self.__is_shutdown:
            self._log.info("Shutting down SMQTK Controller")

            # Run through active ASW processes, interrupting them
            for asw in self._state_to_worker.values():
                asw.interrupt()

            # Sending terminal packets
            self.vcd_controller.queue(None)
            self.ecd_controller.queue(None)

            # Joining sub-controllers
            self._log.info("Joining sub-controllers. This instance is now "
                           "defunct.")
            self.vcd_controller.join()
            self.ecd_controller.join()

            self.__is_shutdown = True
        else:
            self._log.info("SmqtkController already shutdown.")

    ###
    ### IQR Search API
    ###

    def get_search_sessions(self):
        """
        :return: Tuple of initialized IRQ session UUIDs
        :rtype: tuple of uuid.UUID

        """
        with self._iqr_search_sessions_lock:
            return self._iqr_search_sessions.keys()

    def get_iqr_search_state(self, search_uuid):
        """
        :param search_uuid: the UUID of a search
        :type search_uuid: uuid.UUID
        :return: The current iqr search state for the given UUID
        :rtype: RefinableSearchState

        :raises ValueError: Invalid parameter type specified (UUID)
        :raises KeyError: Given search UUID does not match an existing started
            search.

        """
        if not isinstance(search_uuid, uuid.UUID):
            raise ValueError("Not given a valid UUID for the search uuid!")
        with self._iqr_search_sessions_lock:
            return self._iqr_search_sessions[search_uuid]

    def get_archive_search_state(self, search_uuid, state_uuid):
        """
        :param search_uuid: the UUID of a search
        :type search_uuid: uuid.UUID
        :return: The current archive search state for the given UUID
        :rtype: RefinableSearchState

        :raises ValueError: Invalid parameter type specified (UUID)
        :raises KeyError: Given search UUID does not match an existing started
            search.

        """
        if not isinstance(search_uuid, uuid.UUID):
            raise ValueError("Not given a valid UUID for the search uuid!")
        with self._archive_search_map_lock:
            return self._archive_search_map[search_uuid, state_uuid]

    def init_new_search_session(self, event_type, search_query,
                                iqr_distance_kernel, state_uuid=None):
        """
        Initialize a new IQR-enabled search session, returning the UUID
        referring to this session. The given distance kernel **must** support
        sub-matrix extraction (symmetric matrix).

        Initial zero-shot results are computed external to from the backend, so
        all we do here is initialize things. No async processing is spawned by
        calling this method.

        :param event_type: The event type goal of this search query
        :type event_type: int or None
        :param search_query: The query of the search being performed
        :type search_query: str
        :param iqr_distance_kernel: The video distance kernel to use with this
            search during IQR.
        :type iqr_distance_kernel: DistanceKernel
        :return: A new UUID object that doesn't conflict with other IQR UUIDs
        :rtype: uuid.UUID

        """
        # zero-shot already performed on the client/server-size (not in backend)
        # No refinement yet at this point.

        # Creating initial search state and creating search session slot.
        init_state = RefinableSearchState(event_type, search_query,
                                          iqr_distance_kernel,
                                          self._default_classifier_config,
                                          self._mdb_info, self._result_mid, state_uuid = state_uuid)
        with self._iqr_search_sessions_lock:
            self._iqr_search_sessions[init_state.search_uuid] = init_state

        HWSessionInitialize(init_state.search_uuid, iqr_distance_kernel)

        return init_state.search_uuid

    def set_iqr_search_state(self, search_uuid, new_state):
        """
        Side-effects:
            - interrupts processing associated with the current state for the
              given search UUID, if there is search worker associated at all.

        :raises ValueError: Invalid parameter type (UUID, state)
        :raises KeyError: Given search UUID does not match an existing stated
            search.

        :param search_uuid: UUID of the search in question
        :type search_uuid: uuid.UUID
        :param new_state: the new state for the search. This state should be
            constructed with the previous state.
        :type new_state: RefinableSearchState

        """
        if not isinstance(search_uuid, uuid.UUID):
            raise ValueError("Not given a valid UUID for the search uuid!")
        if not isinstance(new_state, RefinableSearchState):
            raise ValueError("Not given a valid search state object!")
        with self._iqr_search_sessions_lock:
            if search_uuid not in self._iqr_search_sessions:
                raise KeyError(search_uuid)

            cur_state = self._iqr_search_sessions[search_uuid]

            # Interrupt any current processing associated with the current state
            # before we assign a new current state to the search UUID.
            if self._state_to_worker.get(cur_state):
                self._log.info("Interrupting processing in previous state")
                self._state_to_worker[cur_state].interrupt()

            self._iqr_search_sessions[search_uuid] = new_state

    def refine_iqr_search(self, search_uuid,
                          refined_positive_ids, refined_negative_ids,
                          removed_positive_ids, removed_negative_ids):
        """
        Refine an existing IQR session given the session UUID and refinement
        data (user-specified video positive/negative feedback).

        :param search_uuid: UUID of the session to refine.
        :type search_uuid: uuid.UUID or str
        :param refined_positive_ids: video ids of new positive videos
        :type refined_positive_ids: Iterable of int
        :param refined_negative_ids: video ids of new negative videos
        :type refined_negative_ids: Iterable of int
        :param removed_positive_ids: IDs that were previously labeled positive
            but are now not labeled as such. This does not include IDs that have
            been selected as negatives, i.e. in ``refined_negative_ids``.
        :type removed_positive_ids: Iterable of int
        :param removed_negative_ids: IDs that were previously labeled negative
            but are now not labeled as such. This does not include IDs that have
            been selected as positives, i.e. in ``refined_positive_ids``.
        :type removed_negative_ids: Iterable of int

        :return: The database connection info object and the model ID of the
            results within that location to consider. This location will be
            update periodically with results from classification work.
        :rtype: (DatabaseInfo, str)

        """
        if isinstance(search_uuid, types.StringTypes):
            search_uuid = uuid.UUID(search_uuid)

        assert isinstance(search_uuid, uuid.UUID), \
            "[%s] Required a UUID object as a session ID!" % search_uuid

        # Create new state for this step, taking into account previous state
        with self._iqr_search_sessions_lock:
            self._log.info("[%s] Creating new search state from previous state",
                           search_uuid)
            cur_state = \
                RefinableSearchState(self.get_iqr_search_state(search_uuid))

            for vID in refined_positive_ids:
                cur_state.register_positive_feedback(vID)
            for vID in refined_negative_ids:
                cur_state.register_negative_feedback(vID)
            for vID in removed_positive_ids:
                cur_state.remove_positive_feedback(vID)
            for vID in removed_negative_ids:
                cur_state.remove_negative_feedback(vID)

            # Set and start async search process
            self._log.info("[%s] Setting new search state -> [%s]",
                           search_uuid, cur_state.uuid)

            self.set_iqr_search_state(search_uuid, cur_state)

            # Create worker process object and set to map
            asw = AsyncSearchWorker(cur_state, self.ecd_controller, True)
            self._state_to_worker[cur_state] = asw
            self._state_to_worker[cur_state].start()

            # Not using get_search_dbinfo to prevent more use of the RLock for
            # efficiency. This is equivalent to making that call, but with out
            # additional checks that, at this point, can be guaranteed.
            return cur_state.mdb_info, cur_state.result_mID

    def archive_search(self, search_uuid, state_uuid, distance_kernel):
        """
        Use the IQR training state of the given search session on a separate
        video set defined by a given distance kernel. This will interrupt any
        existing archive search process running

        :raises ValueError: If the given state UUID doesn't match up to a state
            currently associated with the given search session UUID.

        :param search_uuid: UUID of the search session
        :type search_uuid: uuid.UUID or str
        :param state_uuid: The UUID of the IQR state to use as the base of this
            archival search.
        :type state_uuid: uuid.UUID
        :param distance_kernel: The distance kernel defining the other data set
        :type distance_kernel: DistanceKernel
        :return: The database connection info object and the model ID of the
            results within that location to consider. This location will be
            update periodically with results from classification work.
        :rtype: (DatabaseInfo, str)

        """
        if isinstance(search_uuid, types.StringTypes):
            search_uuid = uuid.UUID(search_uuid)
        assert isinstance(search_uuid, uuid.UUID), \
            "[%s] Required a UUID object as a session ID!" % search_uuid

        if isinstance(state_uuid, types.StringTypes):
            state_uuid = uuid.UUID(state_uuid)
        assert isinstance(state_uuid, uuid.UUID), \
            "[%s] Required a UUID object as a state ID!" % state_uuid

        with self._archive_search_map_lock:
            self._log.info("[%s] Starting search using current state over "
                           "given dataset", search_uuid)

            # Find the IQR state to use by UUID. If not found
            iqr_state = None
            runner_state = self.get_iqr_search_state(search_uuid)
            while (runner_state is not None) and (iqr_state is None):
                if runner_state.uuid == state_uuid:
                    iqr_state = runner_state
                runner_state = runner_state.parent_state

            if iqr_state is None:
                raise ValueError("[%s] No state by the given search UUID is "
                                 "associated with the given search session id "
                                 "%s"
                                 % (state_uuid, search_uuid))

            # Create a new state and then duck-punch it with current IQR state
            # details.
            archive_state = RefinableSearchState(iqr_state.search_event_type,
                                                 iqr_state.search_query,
                                                 iqr_state.distance_kernel,
                                                 iqr_state.classifier_config,
                                                 iqr_state.mdb_info,
                                                 iqr_state.result_mID)
            archive_state.register_positive_feedback(iqr_state.positives)
            archive_state.register_negative_feedback(iqr_state.negatives)
            # Force override state search session UUID with the current on as
            # this method of construction creates a new UUID, which we don't
            # want
            archive_state._search_uuid = iqr_state.search_uuid

            self._archive_search_map[search_uuid, state_uuid] = archive_state

            # Create new, custom ASW with training turned off.
            asw = AsyncSearchWorker(archive_state, self.ecd_controller, False,
                                    distance_kernel)
            self._state_to_worker[archive_state] = asw
            self._state_to_worker[archive_state].start()

            return archive_state.mdb_info, archive_state.result_mID

    def rollback_iqr_search_state(self, search_uuid, rollback_level=1):
        """
        Rollback a search chain by N states where N is the value given to
        ``rollback_level`` (1 by default).

        :param search_uuid:
        :type search_uuid:
        :param rollback_level:
        :type rollback_level:

        """
        current_state = self.get_iqr_search_state(search_uuid)
        if current_state.num_parents < rollback_level:
            raise ValueError("Rollback level exceeded available state in the "
                             "sequence!")

        # Stop/Kill any processing that is currently active associated with the
        # current search state.

        # poll back N levels and reinstate that search state as the current
        # state in the session map (maintains child references until they are
        # overwritten with a refinement on this rollback state).
        rollback_state = current_state
        for _ in xrange(rollback_level):
            # noinspection PyUnresolvedReferences
            # reason -> rollback level amount checked above. Shouldn't underflow
            #           if check passed.
            rollback_state = rollback_state.parent_state

        # noinspection PyTypeChecker
        # reason -> None checked for above.
        self.set_iqr_search_state(rollback_state.search_uuid, rollback_state)

        # TODO: Create ASW for state and start processing!

    def iqr_search_status(self, search_uuid):
        """

        Description taken from ``AsyncSearchWorker.get_pool_results()``
        ---------------------------------------------------------------

        Return indexing progress per pool.

        If the search runtime has not yet been started, then an empty tuple is
        returned. If the runtime has started, but no work indexing has finished
        for a pool size, the associated value for the pool size will be None. If
        an error occurred during the runtime, the final element of the returned
        structure will be a tuple detailing the error that occurred (structure
        detailed below.

        Structure:
        - Normally, tuple of N collections where N is the number of pools in the
        schedule
            - each entry will look like:
                (pool_size<int>, indexed_ids<set or None>)
            - If the indexed IDs iterable is None, then that means the
                indexer has yet to complete for that pool size and ranking
                processing has NOT been queued.
            - Each pool of indexed IDs is additive upon previous indexed ID
                pools. i.e. the ``indexed_ids`` set for pool M is the union of
                this set with all other sets for pool sizes < M.
        - If an error occurred during the run of this ASW, an extra entry will
          be included at the end of the status tuple whose first index value
          is the string "error" (look at structure index ``ret[-1][0]``)
            - the value associated with this error entry, if it exists, will be
              a tuple of two values. The first is a string describing the
              exception that was thrown, and the second is the traceback string
              that was associated with the exception.

        An example return format follows:

        (
          (10, frozenset([1, 2, 4])),
          (50, frozenset([6, 7, 9])),
          (100, None),
          ... ,
          ('error', (<str message>, 'Traceback:...'))
        )

        Sets returned for a pool size are additive upon previous pool size sets.
        In the above example, this means that, for pool size 50, the full set of
        indexed IDs would be [1,2,4,6,7,9], not just [6,7,9]. This also means
        that the set returned for pool size 100 (and higher), when indexing
        finishes, would at least include [1,2,4,6,7,9].

        Remember that the tuple may be empty and my not contain an error entry
        if everything went smoothly (structure[-1][0] != 'error').


        :raises KeyError: Given UUID doesn't refer to an active search.

        :param search_uuid: The UUID of the search to get results from for the
            current state
        :type search_uuid: uuid.UUID

        """
        with self._iqr_search_sessions_lock:
            return \
                self._state_to_worker[self.get_iqr_search_state(search_uuid)] \
                    .get_pool_status()

    def archive_search_status(self, search_uuid, state_uuid):
        """
        Get the search status for the current archive search for the given
        search uuid.

        :param search_uuid: UUID of the search session
        :type search_uuid: uuid.UUID or str
        :param state_uuid: The UUID of the IQR state to use as the base of this
            archival search.
        :type state_uuid: uuid.UUID
        :return:
        :rtype:

        """
        return \
            self._state_to_worker[self.get_archive_search_state(search_uuid,
                                                                state_uuid)]\
            .get_pool_status()

    # TODO: Async state saving mechanism
    #       - If system crashes, power outage, etc. we are able to restart the
    #           backend in a near identical state at the time of the
    #           crash/shutdown.
    #       - Other transportation reasons.
    #       - Probably to work on a python thread instead of a separate process


if __name__ == '__main__':
    import logging
    logging.basicConfig()
    import doctest
    doctest.testmod()
