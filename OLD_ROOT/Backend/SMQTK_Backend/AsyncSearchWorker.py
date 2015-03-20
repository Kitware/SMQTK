"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""


import cPickle
import numpy as np
import os
import os.path as osp
import pymongo
import signal
import time
import traceback

from .ControllerProcess import (
    ControllerProcess
)
from .ECDQueuePacket import ECDQueuePacket, ECDInterruptAgentPacket
from .HWIndexerInterface import HWIndex
from .utils.SignalHandler import SignalHandler

from EventContentDescriptor.iqr_modules import iqr_model_train

from FeatureMemory import DummyRWLock

class AsyncSearchWorker (ControllerProcess):

    def __init__(self, search_state, ecd_controller, do_training,
                 distance_kernel_override=None):
        """
        Creates an AsyncSearchWorker instance. Uses a database for inter-process
        communication of data structures.

        :param search_state: A search state to work with
        :type search_state: RefinableSearchState
        :param ecd_controller: The ECDController to fork work off to.
        :type ecd_controller: ECDController
        :param do_training: Whether this worker should do training first over
            the given state's data and adjudication before proceeding to
            ranking.
        :type do_training: bool
        :param distance_kernel_override: Using this kernel instead of the kernel
            in the given search state.
        :type distance_kernel_override: DistanceKernel

        """
        super(AsyncSearchWorker, self).__init__("AsyncSearchWorker")

        # The ECDController object to manage ECD processing work via
        # work packets
        #: :type: ECDController
        self._ecd_controller = ecd_controller

        # The search state that will act as the filter with which we
        # treat the ECDController (uuid contexts)
        #: :type: RefinableSearchState
        self._search_state = search_state

        self._do_training = do_training
        self._distance_kernel_override =distance_kernel_override

        # Database connection info base for where we will store our IPC stuff
        # Doc format:
        # { asw_uuid: <str_uuid>,
        #   pool_size: <int>,
        #   indexed_ids: <pickled_None_or_set> }
        self._db_client = pymongo.MongoClient(self._search_state.mdb_info.host,
                                              self._search_state.mdb_info.port)

        # Seed database IPC element with initial values
        #coll = self._get_ipd_collection()
        #for pool_size in self._search_state.pool_size_schedule:
        #    doc = {'asw_uuid': str(self.uuid), 'pool_size': pool_size,
        #           'indexed_ids': cPickle.dumps(None)}
        #    coll.insert(doc)

        # Signal handler to control interrupts on the runtime to ensure a safe
        # shutdown.
        self._sh = SignalHandler()

    def _get_ipc_collection(self):
        """
        :return: MongoDB connection and cursor to our IPC collection. The client
            returned should be closed when done.

        """
        db = self._db_client[self._search_state.mdb_info.name]
        coll = db["ASW-%s" % self.uuid]
        return coll

    @property
    def is_learning(self):
        """
        :return: Whether this process is performing training or not or not.
        :rtype: bool
        """
        return self._do_training

    #noinspection PyAttributeOutsideInit
    def _run(self):
        task_uuid = self._search_state.uuid

        # Handler method for protecting against multiple SIGINT signals,
        # ensuring only one getting through to allow a shutdown that is not
        # interruptable.
        self._do_interrupt = True

        def interrupt_handle(signum, _):
            if self._do_interrupt:
                # pretend we didn't catch it
                self._sh.reset_signal(signum)
                self._do_interrupt = False
                raise KeyboardInterrupt("Raised from signal handler")

        self._sh.register_signal(2, interrupt_handle)  # SIGINT registration
        coll = self._get_ipc_collection()

        # Decide which distance kernel to use
        if self._distance_kernel_override:
            dk = self._distance_kernel_override
        else:
            dk = self._search_state.distance_kernel

        # we don't want the distance kernel to change while we perform
        # [learning and] ranking using its properties. Releasing in finally
        # of try/catch after workers have finished / error occurred.
        dk.get_lock().acquireRead()

        search_start = time.time()
        try:
            self._log.info("[%s] Starting AsyncSearchWorker runtime (pid: %d)",
                           self._search_state.uuid, self.pid)

            if self._do_training:
                self._log.info("[%s] Performing training for given state",
                               task_uuid)
                idx2cid_map, idx_is_bg, m = \
                    dk.symmetric_submatrix(*self._search_state.positives)
                # The inverse of idx_is_bg, where True indicated and index that
                # corresponds to a truth video.
                labels_train = tuple(not b for b in idx_is_bg)

                # Determine model files for saving
                model_fpath = osp.join(self._ecd_controller.work_dir,
                                       'model.%s' % task_uuid)
                svIDs_fpath = osp.join(self._ecd_controller.work_dir,
                                       'svids.%s' % task_uuid)

                self._log.info("[%s] making training call", task_uuid)
                # print "[%s] mat: %s\n" \
                #       "     labels_train: %s\n" \
                #       "     idx2cid: %s" \
                #       % (task_uuid, m, np.array(labels_train), idx2cid_map)
                self._log.info("Type of m is %s"%str(type(m)))
                self._log.info("Type of idx2cid_map is %s"%str(type(idx2cid_map)))

                ret_dict = iqr_model_train(model_fpath, m,
                                           np.array(labels_train), idx2cid_map)

                # Saving SV ID list and creating overloaded "model path"
                with open(svIDs_fpath, 'w') as svIDs_file:
                    self._log.info("[%s] Saving support vector IDs: "
                                   "(count: %d) %s",
                                   task_uuid, len(ret_dict['clipids_SVs']),
                                   svIDs_fpath)
                    cPickle.dump(ret_dict['clipids_SVs'], svIDs_file)
                overloaded_model_path = '::'.join((model_fpath, svIDs_fpath))

                self._log.info("[%s] Modifying classifier config for state",
                               task_uuid)
                c_config = self._search_state.classifier_config
                c_config[str(self._search_state.search_event_type)] = {
                    "classifiers": {},
                    "fusion": overloaded_model_path
                }
                self._search_state.classifier_config = c_config

            # Updating "pool" information in IPC database with all clip IDs.
            # SVM ranking algorithm considers the column IDs as the field to
            # rank over (handles case where given dk isn't symmetric; row and
            # col ID sets differ).
            to_rank_cIDs = frozenset(dk.col_id_map())
            query = {'asw_uuid': str(self.uuid), 'pool_size': len(to_rank_cIDs)}
            update = {'$set': {'indexed_ids': cPickle.dumps(to_rank_cIDs)}}
            coll.update(query, update, upsert=True)

            self._log.info("[%s] forming and sending work packet.", task_uuid)
            packet = ECDQueuePacket(task_uuid, 'search',
                                    self._search_state.search_event_type,
                                    to_rank_cIDs, dk,
                                    self._search_state.classifier_config,
                                    self._search_state.positives,
                                    self._search_state.negatives,
                                    str(task_uuid),
                                    self._search_state.result_mID)
            self._log.info("Done forming packet")
            self._ecd_controller.queue(packet)

            ####################################################################
            #### For use with indexing (dummy processing)
            #
            ## WARNING: Since we are queueing these under the same requester ID
            ## work is effectively happening in a synchronous manner within the
            ## ECDC. i.e. work on the second pool's clips can't start until the
            ## first pool's clips are done. This seems to defeat the point of
            ## parallelism...
            #
            ## Loop through pool sizes, indexing them
            #ranked_ids = set()  # keeps track of IDs seen
            #for pool_size in self._search_state.pool_size_schedule:
            #    self._log.info("[%s] Starting work for pool: %s",
            #                   task_uuid, pool_size)
            #
            #    # index to get pool to rank
            #    self._log.info("[%s] Indexing pool", task_uuid)
            #    indexed_ids = HWIndex(self._search_state.search_uuid, pool_size,
            #                          self._search_state.positives,
            #                          self._search_state.negatives)
            #    #: :type: set
            #    indexed_ids = set(indexed_ids).difference(ranked_ids)
            #    self._log.debug("[%s] num indexed ids: %i",
            #                    task_uuid, len(indexed_ids))
            #
            #    # Setting indexed ids to
            #    query = {'asw_uuid': str(self.uuid), 'pool_size': pool_size}
            #    update = {'$set': {'indexed_ids': cPickle.dumps(indexed_ids)}}
            #    coll.update(query, update)
            #
            #    # fire off ranking job
            #    self._log.info("[%s] Firing off ECDController work packet",
            #                   task_uuid)
            #    packet = ECDQueuePacket(task_uuid, 'search',
            #                            self._search_state.search_event_type,
            #                            indexed_ids,
            #                            self._search_state.classifier_config,
            #                            self._search_state.positives,
            #                            self._search_state.negatives,
            #                            str(task_uuid),
            #                            self._search_state.result_mID)
            #    self._ecd_controller.queue(packet)
            #
            #    ranked_ids.update(indexed_ids)
            #
            #    # DUMMY: Sleeping in between pool processing steps to simulate
            #    # stuff actually happening through out the system
            #    time.sleep(5.0)

            # add terminal packet for when they're done
            self._log.info("[%s] Sending terminal work packet to allow jobs to "
                           "finish.", task_uuid)

            dk._rw_lock = DummyRWLock()
            self._ecd_controller.queue(ECDQueuePacket(task_uuid, 'search',
                                                      self._search_state
                                                          .search_event_type,
                                                      None, dk,
                                                      self._search_state
                                                          .classifier_config))

            all_complete = False
            while not all_complete:
                worker_infos = self._ecd_controller.get_worker_info(task_uuid)
                # Queue packet may not have reached ECDC yet, so we should only
                # check progress when the ECDC has registered workers.
                if worker_infos:
                    exit_codes = [w.exitcode for w in worker_infos]
                    if None not in exit_codes:  # alive proc exit code is None
                        all_complete = True
                        # Report error if some of the exit codes for processes
                        # were non-zero.
                        if any(exit_codes):  # i.e. if anything not 0
                            raise RuntimeError("Not all workers exited with a "
                                               "0 exit code!")
                time.sleep(0.1)

            #noinspection PyAttributeOutsideInit
            self._do_interrupt = False

        except KeyboardInterrupt:
            self._log.info("[%s] Interrupted!", task_uuid)

            # Send an interrupt packet regardless, since if nothing was actually
            # send to the ECDC yet, this does nothing.
            self._ecd_controller.queue(ECDInterruptAgentPacket(task_uuid))

        except Exception, ex:
            self._do_interrupt = False

            # Catch any other exception for propagation and reporting
            self._log.error("[%s] %s: %s",
                            task_uuid, ex.__class__.__name__, str(ex))
            self._log.error("[%s] exception traceback:\n%s",
                            task_uuid, traceback.format_exc())

            tb_str = traceback.format_exc()

            if coll is None:
                coll = self._get_ipc_collection()

            query = {'asw_uuid': str(self.uuid), 'pool_size': 'error'}
            # Stringifying the exception as storing the exception directly can
            # cause pickle/json errors depending the exact exception that was
            # thrown.
            ex_str = "%s: %s" % (type(ex).__name__, str(ex))
            update = {'$set': {'indexed_ids': cPickle.dumps((ex_str, tb_str))}}
            coll.update(query, update, upsert=True)

        finally:
            dk.get_lock().releaseRead()
            self._db_client.close()
            # TODO: Clean up this run's model files?
            self._log.info("[%s] Done (%f s)", task_uuid,
                           time.time() - search_start)

    def join(self, timeout=None):
        self._db_client.close()

    def get_pool_status(self):
        """
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

        :return: The above described tuple structure
        :rtype: tuple of (int, set of int or (exceptions.Exception, str))

        """
        with self._db_client:
            coll = self._get_ipc_collection()
            query = {'asw_uuid': str(self.uuid)}
            ret_docs = coll.find(query)  # length equals number of pool sizes
            pool_progress = []
            for doc in ret_docs:
                pool_progress.append((doc['pool_size'],
                                      cPickle.loads(str(doc['indexed_ids']))))
            # return in order of pool size (lowest first)
            return tuple(sorted(pool_progress))

    def interrupt(self):
        """
        Interrupt current processing occurring in the ECDController system
        related to this worker, and then interrupt ourselves.

        """
        self._log.info("[%s] Interrupting self", self._search_state.uuid)
        try:
            os.kill(self.pid, signal.SIGINT)
        except OSError, ex:
            # Ignore the "no such process" error. Others should pass through.
            if ex.errno != 3:
                raise
        self.join()
