"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'paul.tunison'

import logging
from nose import tools
import numpy as np
import pymongo
import os
import os.path as osp
import pymongo
import random
import tempfile
import time
import unittest
import uuid

from SMQTK_Backend.FeatureMemory import \
    initFeatureManagerConnection, \
    getFeatureManager
from SMQTK_Backend.SmqtkController import SmqtkController
from SMQTK_Backend.RefinableSearchState import RefinableSearchState
from SMQTK_Backend.VCDController import VCDController


# noinspection PyPep8Naming
class test_SmqtkController (unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super(test_SmqtkController, self).__init__(methodName)
        self.log = logging.getLogger('.'.join((self.__module__,
                                               self.__class__.__name__)))
        self.poll_interval = 0.1

    @classmethod
    def setUpClass(cls):
        # local FeatureManager server may have already been initialized by
        # another unittest
        try:
            initFeatureManagerConnection()
        except ValueError:
            pass

    def setUp(self):
        # uses default database location (localhost, 27017)
        gc_config = SmqtkController.generate_config()

        # Modifying VCDController database location so we can safely remove it
        # when we're done
        self._vcd_store_loc = osp.join(osp.dirname(osp.abspath(__file__)),
                                       'work/vcd_store')
        gc_config.set(VCDController.CONFIG_SECT, 'store_name',
                      self._vcd_store_loc)

        # TODO: change the smqtk_controller:mongo_database
        self.gc = SmqtkController(gc_config)

        data_dir = osp.abspath(osp.join(osp.dirname(__file__), 'data'))
        print "Data dir:", data_dir

        symmetric_clipids = np.loadtxt(osp.join(data_dir,
                                                'symmetric_clipids.txt'))
        symmetric_bgflags = np.loadtxt(osp.join(data_dir,
                                                'symmetric_bgflags.txt'))
        # noinspection PyCallingNonCallable
        symmetric_dk_mat = \
            np.matrix(np.load(osp.join(data_dir,
                                       'symmetric_distance_kernel.npy')))
        symmetric_bgclipids = np.array([symmetric_clipids[i]
                                        for i, e in enumerate(symmetric_bgflags)
                                        if e])
        # noinspection PyUnresolvedReferences
        #: :type: DistanceKernel
        self.dk_iqr = getFeatureManager().DistanceKernel(symmetric_clipids,
                                                         symmetric_clipids,
                                                         symmetric_dk_mat,
                                                         symmetric_bgclipids)

        asymmetric_row_ids = np.loadtxt(osp.join(data_dir,
                                                 'asymmetric_clipids_rows.txt'))
        asymmetric_col_ids = np.loadtxt(osp.join(data_dir,
                                                 'asymmetric_clipids_cols.txt'))
        # noinspection PyCallingNonCallable
        asymmetric_dk_mat = \
            np.matrix(np.load(osp.join(data_dir,
                                       'asymmetric_distance_kernel.npy')))
        # noinspection PyUnresolvedReferences
        #: :type: DistanceKernel
        self.dk_arc = getFeatureManager().DistanceKernel(asymmetric_row_ids,
                                                         asymmetric_col_ids,
                                                         asymmetric_dk_mat)

    def tearDown(self):
        self.gc.shutdown()
        # removing vcd_store file created as by-product
        os.remove(self._vcd_store_loc)

    def test_defaults(self):
        self.assertTrue(osp.isdir(self.gc.data_directory))
        self.assertEqual(self.gc.work_directory,
                         osp.join(tempfile.gettempdir(),
                                  'SmqtkWork'))

    def test_multiple_shutdown(self):
        self.gc.shutdown()
        self.gc.shutdown()

    def test_session_accessor(self):
        self.assertSequenceEqual(self.gc.get_search_sessions(), ())
        sID = self.gc.init_new_search_session(None, '', self.dk_iqr)
        self.assertIsInstance(sID, uuid.UUID)
        self.assertSequenceEqual(self.gc.get_search_sessions(), (sID,))

    @tools.raises(KeyError)
    def test_get_search_state_fail(self):
        self.gc.get_iqr_search_state(uuid.uuid4())

    def test_get_search_state(self):
        sID = self.gc.init_new_search_session(None, '', self.dk_iqr)
        ss = self.gc.get_iqr_search_state(sID)
        self.assertIsInstance(ss, RefinableSearchState)

    @tools.raises(AssertionError)
    def test_refine_bad_search_uuid(self):
        """ test giving refine a non-UUID value """
        #noinspection PyTypeChecker
        # reason -> the point of the test
        self.gc.refine_iqr_search(None, (), (), (), ())

    @tools.raises(KeyError)
    def test_refine_invalid_search_uuid(self):
        """
        Test when refine given a bad (incorrect) search UUID
        """
        # Nothing is initialized, so this UUID can not be registered.
        self.gc.refine_iqr_search(uuid.uuid4(), (), (), (), ())

    def test_single_cycle_refine(self):
        self.log.info("Initializing search session")
        sID = self.gc.init_new_search_session(None, 'dummy', self.dk_iqr)
        self.log.info("Kicking off refine")
        dbInfo, fMID = self.gc.refine_iqr_search(sID, (1,), (2,), (3,), (4,))
        self.log.info("DatabaseInfo: %s", dbInfo)
        self.log.info("fusion MID  : %s", fMID)

        self.log.info("Waiting for results to populate")
        client = pymongo.MongoClient(dbInfo.host, dbInfo.port)
        coll = client[dbInfo.name][dbInfo.collection]
        results_finished = False
        while not results_finished:
            time.sleep(self.poll_interval)
            progress = self.gc.iqr_search_status(sID)
            clips_done = set(doc['clip_id'] for doc
                             in coll.find({'model_id': fMID}))

            pool_completions = []
            for p_size, indexed_ids in progress:
                if indexed_ids:
                    pool_ids_done = len(indexed_ids.intersection(clips_done))
                    pool_completions.append(
                        float(pool_ids_done) / len(indexed_ids))
                else:
                    pool_completions.append(0.0)

            self.log.info("Pool completions: %s", pool_completions)
            completion = sum(pool_completions) / len(pool_completions)
            self.log.info("Total completion: %s%%", completion * 100)
            results_finished = int(completion)
        self.log.info("Complete")

    def test_single_cycle_from_string_uuid(self):
        """
        Test performing a refine giving it a string UUID
        """
        self.log.info("Initializing search session")
        sID = self.gc.init_new_search_session(None, 'dummy', self.dk_iqr)
        str_uuid = str(sID)
        self.log.info("String UUID: '%s'", str_uuid)
        self.log.info("Kicking off refine")
        dbInfo, fMID = self.gc.refine_iqr_search(str_uuid,
                                                 (1,), (2,), (3,), (4,))
        self.log.info("DatabaseInfo: %s", dbInfo)
        self.log.info("fusion MID  : %s", fMID)

        self.log.info("Waiting for results to populate")
        client = pymongo.MongoClient(dbInfo.host, dbInfo.port)
        coll = client[dbInfo.name][dbInfo.collection]
        results_finished = False
        while not results_finished:
            time.sleep(self.poll_interval)
            progress = self.gc.iqr_search_status(sID)
            clips_done = set(doc['clip_id'] for doc
                             in coll.find({'model_id': fMID}))

            pool_completions = []
            for p_size, indexed_ids in progress:
                if indexed_ids:
                    pool_ids_done = len(indexed_ids.intersection(clips_done))
                    pool_completions.append(
                        float(pool_ids_done) / len(indexed_ids))
                else:
                    pool_completions.append(0.0)

            self.log.info("Pool completions: %s", pool_completions)
            completion = sum(pool_completions) / len(pool_completions)
            self.log.info("Total completion: %s%%", completion * 100)
            results_finished = int(completion)
        self.log.info("Complete")

    @tools.timed(60.0)  # really high end for finishing...
    def test_refine_interruptions_fast(self):
        """
        Test multiple quick refine attempts at random quick intervals
        (sub-second)
        """
        s = time.time()
        n_iters = 10
        dbInfo = fMID = None

        self.log.info("Initializing search session")
        sID = self.gc.init_new_search_session(None, 'dummy', self.dk_iqr)
        for i in xrange(n_iters):
            self.log.info("Refine %d", i+1)
            dbInfo, fMID = self.gc.refine_iqr_search(sID, (i,), (1000000-i,),
                                                     (), ())
            r = random.random()
            time.sleep(random.random())
            self.log.info("Waiting %ss", r)

        self.log.info("Waiting for results to populate")
        client = pymongo.MongoClient(dbInfo.host, dbInfo.port)
        coll = client[dbInfo.name][dbInfo.collection]
        results_finished = False
        while not results_finished:
            time.sleep(self.poll_interval)
            progress = self.gc.iqr_search_status(sID)
            clips_done = set(doc['clip_id'] for doc
                             in coll.find({'model_id': fMID}))

            pool_completions = []
            for p_size, indexed_ids in progress:
                if indexed_ids:
                    pool_ids_done = len(indexed_ids.intersection(clips_done))
                    pool_completions.append(
                        float(pool_ids_done) / len(indexed_ids)
                    )
                else:
                    pool_completions.append(0.0)

            self.log.info("Pool completions: %s", pool_completions)
            completion = sum(pool_completions) / len(pool_completions)
            self.log.info("Total completion: %s%%", completion * 100)
            results_finished = int(completion)
        self.log.info("Complete")
        self.log.info("Time to complete: %s", time.time() - s)

    def test_concurrent_refines(self):
        """
        Test handling multiple refines happening concurrently
        """
        n_concur = 5
        #: :type: dict of (uuid.UUID, (DatabaseInfo, str))
        search_map = {}
        #: :type: dict of (uuid.UUID, float)
        search_progress = {}

        for i in xrange(n_concur):
            sID = self.gc.init_new_search_session(None, chr(ord('a') + i),
                                                  self.dk_iqr)
            search_map[sID] = \
                self.gc.refine_iqr_search(sID,
                                          [int(max(self.dk_iqr.col_id_map())) - i],
                                          [i],
                                          (), ())
            search_progress[sID] = 0.0

        # we know, for this test scenario, the database target will be the same
        # for all
        #noinspection PyUnboundLocalVariable
        #: :type: DatabaseInfo
        ref_dbInfo = search_map[sID][0]
        client = pymongo.MongoClient(ref_dbInfo.host, ref_dbInfo.port)

        results_finished = False
        while not results_finished:
            time.sleep(self.poll_interval)
            self.log.info("###############################################")
            for sID in search_map:
                dbInfo, fMID = search_map[sID]
                pools = self.gc.iqr_search_status(sID)
                coll = client[dbInfo.name][dbInfo.collection]
                clips_done = set(doc['clip_id'] for doc
                                 in coll.find({'model_id': fMID}))

                if pools:
                    assert pools[-1][0] != 'error', \
                        "[%s] An error occurred during processing: %s\n%s" \
                        % (sID, pools[-1][1][0], pools[-1][1][1])

                pool_completions = []
                for psize, indexed_ids in pools:
                    if indexed_ids:
                        pool_ids_done = \
                            len(indexed_ids.intersection(clips_done))
                        pool_completions.append(
                            float(pool_ids_done) / len(indexed_ids)
                        )
                    else:
                        pool_completions.append(0.0)

                self.log.info('[%s] pool completions: %s',
                              sID, pool_completions)
                # possible race condition where underlying search worker on the
                # refinement hasn't progressed to the point where
                if pool_completions:
                    completion = sum(pool_completions) / len(pool_completions)
                else:
                    completion = 0.0
                self.log.info('[%s] Total completion: %s%%',
                              sID, completion * 100)
                search_progress[sID] = completion

            completion = sum(search_progress.values()) / len(search_progress)
            self.log.info("... Run completion: %s%%", completion * 100)
            results_finished = int(completion)
