"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Control structure for Event Content Descriptor workers (classifiers) and fusion
processes. When search or learn requests come in, with a UUID, new process pools
are generated for that request. Since base ECDWorker processes have a
potentially long initialization period,

"""

import cPickle
import multiprocessing
import os
import os.path as osp
import pymongo
from Queue import Empty as QueueEmpty
import signal
import tempfile

from .ControllerProcess import ControllerProcess
from .ECDQueuePacket import ECDQueuePacket, ECDInterruptAgentPacket
from .ECDStore import ECDStore
from .ECDWorkers.classifiers import get_classifiers
from .ECDWorkers.fusion import IqrDemoFusion
from .utils import DatabaseInfo
from .utils import SafeConfigCommentParser
from .utils import SignalHandler


class ECDController (ControllerProcess):
    """
    Process that manages a dispatch queue for processing requests, spawning
    workers based on known VCDWorkers implementations to produce data that is
    inserted into the registered VCDStore.

    For each learn/search query, a number of worker classifiers are fired off
    as well as a fusion process. Often, agents that are requesting search
    operations of the ECDController care primarily, if not only, about the fused
    scores. These scores may be access within the store by looking for entries
    via their MID (model_id) of "fusion.<event_id#>" and their clip id (the
    two parameters to the store's ``get`` method).

    """

    CONFIG_SECT = 'ecd_controller'

    @classmethod
    def generate_config(cls, config=None):
        """
        Generate and return the configuration for an ECD Controller.

        :param config: And existing configuration object to add to. By default
            we will create a new config object and return it. Else the provided
            config object is modified and returned.
        :type config: SafeConfigCommentParser or None

        :return: A new config object, of the same one provided, with new
            sections/options for this controller.
        :rtype: SafeConfigCommentParser

        """
        if config is None:
            config = SafeConfigCommentParser()

        sect = cls.CONFIG_SECT
        if sect not in config.sections():
            config.add_section(sect,
                               'Options for the ECD process controller')

        config.set(sect, 'work_directory',
                   osp.join(tempfile.gettempdir(), 'ecd_controller_work'),
                   'Directory to use for working files.')
        config.set(sect, 'data_directory',
                   osp.join(osp.abspath(osp.dirname(__file__)), 'data'),
                   'Directory that contains data to read from. Nothing should '
                   'be written to this directory.')

        # Runtime related
        config.set(sect, 'runtime_update_interval', '0.5',
                   'Maximum time the runtime will wait on the work queue '
                   'before cycling with a void to update members.')
        config.set(sect, 'worker_join_timeout', '10.0',
                   'Time to wait while joining worker processes during '
                   'shutdown. If this is exceeded when waiting for a worker to '
                   'shutdown we will forcefully interrupt that process.')

        return config

    def __init__(self, config, vcd_controller, db_info):
        """
        :param config: The configuration object for this controller and
            sub-implementations.
        :type config: ConfigParser.ConfigParser
        :param vcd_controller: The VCD Controller so ECD workers may register
            work they need completed.
        :type vcd_controller: VCDController
        :param db_info: The database information to use. We will only consider
            the database host, port and name from the given structure.
        :type db_info: DatabaseInfo

        """
        super(ECDController, self).__init__('ECDController')

        ### Helpers
        # - define 'sect' var before retrieval
        abspath = lambda p: osp.abspath(osp.expanduser(p))
        cget = lambda k: config.get(sect, k)
        #cgetint = lambda k: config.getint(sect, k)
        cgetfloat = lambda k: config.getfloat(sect, k)

        sect = self.CONFIG_SECT
        self._work_dir = abspath(cget('work_directory'))
        self._data_dir = abspath(cget('data_directory'))
        self._runtime_wq_timeout = cgetfloat('runtime_update_interval')
        self._worker_join_timeout = cgetfloat('worker_join_timeout')

        # Make sure work directory exists
        if not osp.isdir(self._work_dir):
            os.makedirs(self._work_dir)

        assert isinstance(db_info, DatabaseInfo)
        self._mdb_info = db_info.copy()
        self._mdb_info.collection = None
        self._mdb_client = pymongo.MongoClient(self._mdb_info.host,
                                               self._mdb_info.port)

        self._default_ecd_store = ECDStore(host=self._mdb_info.host,
                                           port=self._mdb_info.port,
                                           database=self._mdb_info.name)

        self.work_queue = multiprocessing.Queue()
        self._vcdc = vcd_controller

        # DB Shared data structure
        self._agent_worker_info = {}
        col = self._get_ipc_collection()
        col.insert({'ecd_uuid': str(self.uuid),
                    'agent_worker_info':
                    cPickle.dumps(self._agent_worker_info)})

    def _get_ipc_collection(self):
        """
        :return: MongoDB connection and cursor to our IPC collection. The client
            returned should be closed when done.

        """
        db = self._mdb_client[self._mdb_info.name]
        coll = db["ECDC-%s" % self.uuid]
        return coll

    @property
    def work_dir(self):
        return self._work_dir

    def store(self, custom_collection=None):
        if custom_collection:
            if not isinstance(custom_collection, str):
                raise ValueError("The custom collection must be string value!")
            return ECDStore(host=self._mdb_info.host,
                            port=self._mdb_info.port,
                            database=self._mdb_info.name,
                            collection=custom_collection)
        else:
            return self._default_ecd_store

    def queue(self, packet):
        """
        Submit a job message to this controller. A None value given will
        shutdown the controller after current processing completes.

        :param packet: The data packet to queue up.
        :type packet: ECDQueuePacket or None

        """
        assert packet is None or isinstance(packet, ECDQueuePacket), \
            "Not given an ECDQueuePacket to transport!"
        try:
            self.work_queue.put(packet)
            if packet is None:
                self._log.info("Inserted None packet. Runtime will close.")
                # wq close is part of join
        except AssertionError:
            self._log.warning("Failed to insert into a closed work queue! "
                              "Controller must have already been shutdown.")

    def _run(self):
        """
        Runtime loop that waits for work packets to come along.

        When work packets arrive, check if there are already worker processes
        available and running. If not start them. Else given them work.

        The runtime is terminated when given a shutdown signal along the queue.
        This shutdown signal is a simple None object.

        """
        self._log.info("Starting ECD Controller runtime (pid: %d)", self.pid)
        ecd_class_workers = get_classifiers()

        signal_handler = SignalHandler()
        signal_handler.register_signal(signal.SIGINT,
                                       custom_func=lambda *args:
                                       self._log.info("SIGINT caught in ECDC"))

        # Mapping or workers for different input requests. Based on UUID-tagged
        # requests, where requests from the same agent are tagged with the same
        # UUID.
        #
        # {
        #   <UUID>: [<list of worker processes>],
        #   ...
        # }
        #: :type: dict of (uuid.UUID, tuple of ECDWorkerBaseProcess)
        agent_worker_map = {}

        runtime_active = True
        while runtime_active:
            try:
                #: :type: ECDQueuePacket or None
                work_packet = self.work_queue\
                                  .get(timeout=self._runtime_wq_timeout)
            except IOError, ex:
                if ex.errno != 4:
                    raise
                # else its just an interrupt on the get, which is ignorable
                # (only seen this happen due to a ctrl-C event / SIGINT which we
                # are catching)
                self._log.debug("caught io interrupt, ignoring")
                work_packet = 'queue get interrupted'
            except QueueEmpty:
                # get timeout exceeded. Create void packet to cycle through
                # updating members
                work_packet = 'queue.get timeout'

            # self._log.debug("New work packet: <%s>", work_packet)

            if signal_handler.is_signal_caught(signal.SIGINT):
                self._log.info("SIGINT caught. Exiting runtime.")
                runtime_active = False
                continue

            elif work_packet is None:  # shutdown packet
                self._log.info("Safe-shutdown work packet (None) received. "
                               "Exiting runtime.")
                runtime_active = False

            elif isinstance(work_packet, ECDInterruptAgentPacket):
                if work_packet.requester_uuid in agent_worker_map:
                    self._log.info("Interrupt workers under agent [%s]",
                                   work_packet.requester_uuid)
                    for p in agent_worker_map[work_packet.requester_uuid]:
                        self._log.info("Interrupting '%s' -> pid:%d",
                                       p.name, p.pid)
                        try:
                            os.kill(p.pid, signal.SIGINT)
                            self._log.info("'%s' interrupted", p.name)
                        except OSError, ex:
                            # OSError #3 is a "process doesn't exist" exception,
                            # which may safely occur if the process is not alive
                            # anymore due to it shutting down via some other
                            # mechanism, like just finishing normally.
                            # Basically, this error can be ignored.
                            if ex.errno != 3:
                                raise
                            else:
                                self._log.info("'%s' already closed", p.name)
                else:
                    self._log.info("No agent [%s] to interrupt")

            elif isinstance(work_packet, ECDQueuePacket):
                req_uuid = work_packet.requester_uuid

                # Skip if we received a work packet for a new agent with a None
                # value for work clips. Since a None value here is usually the
                # shutdown signal for registered agent workers, if the agent is
                # not registered, it should be a no-op.
                if req_uuid not in agent_worker_map \
                        and work_packet.clips is None:
                    self._log.info("Received an agent shutdown signal for an "
                                   "agent that is not registered. Skipping "
                                   "and waiting for more work.")
                    continue

                # TODO: For roll-back support, also reinitialize processes if
                #       uuid *is* in map, but none of the workers are not alive.
                if req_uuid not in agent_worker_map:
                    # Create a worker map entry and initialize fusion process +
                    # associated workers based on classifier configuration.
                    self._log.debug("work packet revealed new agent UUID. "
                                    "Creating new processes.")

                    # should be one of [ 'learn' | 'search' ]
                    req_type = work_packet.request_type
                    e_type = str(work_packet.event_type)  # str b/c json dict

                    # slot for worker registry
                    workers = []

                    # start up base workers based on configuration for event
                    # type
                    self._log.debug("[agent:%s] initializing base classifier "
                                    "workers", req_uuid)

                    # Dictionary, for an event type, of base classifier types to
                    # configurations mapping
                    classifier_config = work_packet.classifier_config

                    if e_type not in classifier_config:
                        self._log.info("Unrecognized event type, defaulting to "
                                       "general case.")
                        e_type = str(None)

                    classifier_config = classifier_config[e_type]

                    # Initialize base workers
                    cfier_section = classifier_config['classifiers']
                    for cfier_type in cfier_section.keys():
                        # Skip classifier types that start with '_', like a
                        # "_comment" entry or the like.
                        if cfier_type.startswith('_'):
                            continue

                        # Attempt to retrieve the worker implementation class
                        # based on the configuration for the request event type
                        try:
                            cfier_type_class = \
                                ecd_class_workers[cfier_type][req_type]
                        except KeyError, ex:
                            raise ValueError("Invalid configured classifier "
                                             "type %s! No implementations of "
                                             "this type exist!"
                                             % str(ex))

                        self._log.debug("[agent:%s] -- initializing '%s' "
                                        "workers", req_uuid,
                                        cfier_type_class.__name__)

                        # Create the worker classes using the cfier type impl
                        for config in cfier_section[cfier_type]:
                            name = '%s.%s' % (cfier_type, config[0])
                            self._log.debug("[agent:%s]    -- '%s' "
                                            "configuration", req_uuid, name)
                            w = cfier_type_class(name, config[1], config[2],
                                                 self._vcdc, self)
                            workers.append(w)

                    # initialize fusion process
                    # TODO: This is forcing the use of ``SimpleFusion``
                    #       Should probably change this with the official fusion
                    #       thing when the time comes. Or make it configurable?
                    self._log.debug('(agent:%s) initializing fusion worker',
                                    req_uuid)
                    if work_packet.result_MID:
                        fusion_name = work_packet.result_MID
                    else:
                        fusion_name = "fusion.%s" % work_packet.event_type
                    fusion_model = classifier_config['fusion']
                    #f = SimpleAvgFusion(fusion_name, fusion_model,
                    #                    self._vcdc, self,
                    #                    tuple(w for w in workers))
                    f = IqrDemoFusion(fusion_name, fusion_model, self._vcdc,
                                      self, tuple(w for w in workers))
                    workers.append(f)

                    agent_worker_map[req_uuid] = tuple(workers)

                    # Start worker processes
                    self._log.debug("(agent:%s) booting up worker processes",
                                    req_uuid)
                    for w in workers:
                        w.start()

                # agent UUID will, by this point, have workers associated with
                # it, so pass on the work packet. The work packet, at this
                # point, will also be a valid one from a system perspective (aka
                # not a shutdown packet).
                self._log.info('(agent:%s) distributing work packet %s',
                               req_uuid, work_packet)
                if work_packet.clips is None:
                    self._log.info("(agent:%s) None clip, agent workers will "
                                   "shutdown when completed with any "
                                   "outstanding work.",
                                   req_uuid)
                    # Not performing join on workers that are told to shutdown
                    # due to an effect of the multiprocessing system that joins
                    # zombied processes when new processes are spawned.
                for w in agent_worker_map[req_uuid]:
                    w.work_queue.put(work_packet)
            elif work_packet == 'queue.get timeout':
                pass

            else:  # Not a valid packet, just ignore it.
                self._log.debug("Invalid packet received ('%s'). Skipping and "
                                "waiting for next packet.", work_packet)

            # Update IPC data structure
            agent_worker_info = {}
            for agent, workers in agent_worker_map.items():
                agent_worker_info[agent] = tuple(w.get_info() for w in workers)
            coll = self._get_ipc_collection()
            coll.update({'ecd_uuid': str(self.uuid)},
                        {'$set': {'agent_worker_info':
                                  cPickle.dumps(agent_worker_info)}})

        ###
        # Shutdown
        #
        self._log.info("ECDC Runtime shutting down")
        # Send a None to all worker processes across all registered UUID
        # keys to tell them to stop when work is done.
        for agent_uuid in agent_worker_map:
            self._log.info("Sending terminus signal for agent %s workers.",
                           agent_uuid)
            for worker in agent_worker_map[agent_uuid]:
                worker.work_queue.put(None)
        # Now joining **after** letting everyone know this is coming. Doing
        # this allows processes to shutdown/finish-up in parallel instead of
        # doing it one at a time. If processes have non-trivial shutdown
        # procedures, this is more advantageous.
        for agent_uuid in agent_worker_map:
            self._log.info("(agent:%s) Joining worker pool (timeout=%f)",
                           agent_uuid, self._worker_join_timeout)
            for worker in agent_worker_map[agent_uuid]:
                try:
                    self._log.info("(agent:%s) -- joining %s",
                                   agent_uuid, worker)
                    worker.join(self._worker_join_timeout)
                except AssertionError, ex:
                    if ex.message == 'can only join a started process':
                        self._log.debug("(agent:%s) joined not-started process "
                                        "%s", agent_uuid, worker.name)
                    else:
                        raise

                # If the worker is still alive, then timeout must have expired.
                # Terminate process and then join.
                if worker.is_alive():
                    self._log.warn("(agent:%s) worker '%s' process failed join "
                                   "within timeout. Interrupting worker.",
                                   agent_uuid, worker.name)
                    #worker.terminate()
                    try:
                        os.kill(worker.pid, signal.SIGINT)
                    except OSError, ex:
                        # errno 3 is raised when worker isn't alive to send a
                        # signal to, which is perfectly ok given that we're
                        # trying to interrupt the process.
                        if ex.errno != 3:
                            raise
                    worker.join()

        self._mdb_client.close()

        signal_handler.unregister_signal(signal.SIGINT)

    def get_worker_info(self, requester_uuid):
        """
        Get a tuple of ControllerProcessInfo objects for worker processes for a
        given agent ID, or None if no workers are registered for the given UUID.

        :param requester_uuid: UUID of the requester
        :type requester_uuid: uuid.UUID
        :return: Tuple of ControllerProcessInfo objects.
        :rtype: tuple of ControllerProcessInfo

        """
        coll = self._get_ipc_collection()
        #: :type: dict
        agent_worker_info = \
            cPickle.loads(str(
                coll.find_one({'ecd_uuid': str(self.uuid)})['agent_worker_info']
            ))
        return agent_worker_info.get(requester_uuid, None)

    def join(self, timeout=None):
        self.work_queue.close()
        self.work_queue.join_thread()
        self._mdb_client.close()
        super(ECDController, self).join(timeout)

    def interrupt(self):
        self._log.info("Interrupting ECDController processing")
        # Since we want to keep things at least relatively safe, send a SIGINT
        # signal to the runtime. Same effect as sending a None packet, but cuts
        # the queue's line.
        self._log.info("Sending SIGINT")
        os.kill(self.pid, signal.SIGINT)
        self._log.info("Joining self")
        self.join()  # also closes work queue
