# -*- coding: utf-8 -*-

from multiprocessing import Process, current_process, Lock, Queue
import numpy as np
import os
import os.path as osp
import subprocess
import time
from threading import current_thread
import types
import unittest

from SMQTK_Backend.FeatureMemory import (
    get_common_fmm,
    initFeatureManagerConnection,
    removeFeatureManagerConnection,
    getFeatureManager,
    ReadWriteLock,
    DistanceKernel,
    FeatureMemory,
    FeatureMemoryMap
)
from SMQTK_Backend.utils import SimpleTimer, SafeConfigCommentParser


# Because we import the module SimpleTimer instead of the class that the util
# sub-module sets as the symbol when running nosetests? I don't know how this
# happens currently, so this is an interim fix since this is only a test case
# file.
# TODO: Figure out why we import the module here instead of the class when
#       running nosetests.
if isinstance(SimpleTimer, types.ModuleType):
    # noinspection PyUnresolvedReferences
    # -> valid when initial SimpleTimer reference is a module
    SimpleTimer = SimpleTimer.SimpleTimer
if isinstance(SafeConfigCommentParser, types.ModuleType):
    # noinspection PyUnresolvedReferences
    SafeConfigCommentParser = SafeConfigCommentParser.SafeConfigCommentParser


# noinspection PyUnresolvedReferences
class TestFeatureMemory (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # local FeatureManager server may have already been initialized by
        # another unittest
        try:
            initFeatureManagerConnection()
        except ValueError:
            pass

        cls.data_dir = osp.abspath(osp.join(osp.dirname(__file__), 'data'))
        cls.work_dir = osp.abspath(osp.join(osp.dirname(__file__), 'work'))

        cls.sym_cids_file  = osp.join(cls.data_dir, "symmetric_clipids.txt")
        cls.sym_bg_flags   = osp.join(cls.data_dir, "symmetric_bgflags.txt")
        cls.sym_dk_mat_f   = osp.join(cls.data_dir, "symmetric_distance_kernel.npy")
        cls.sym_feat_mat_f = osp.join(cls.data_dir, "symmetric_feature.npy")

    @classmethod
    def tearDownClass(cls):
        removeFeatureManagerConnection()

    # noinspection PyCallingNonCallable
    def setUp(self):
        self.mgr = getFeatureManager()

        self.N = 1024 * 4  # 4096
        self.M = 1024 * 6  # 6144

        # self.N = 1024 * 8
        # self.M = 1024 * 16

        self.id_vec = np.array(range(self.N))
        self.bg_ids = np.array(range(self.N//10))
        self.feature_mat = np.matrix(np.random.random((self.N, self.M)))
        self.dk_mat = np.matrix(np.random.random((self.N, self.N)))

    def test_structure_local_side(self):
        print "===================================="
        print "simple local-side test of structures"
        print "===================================="

        rw_lock = ReadWriteLock()

        #: :type: FeatureMemory
        fm = FeatureMemory(self.id_vec, self.bg_ids, self.feature_mat,
                           self.dk_mat, rw_lock)

        rw2 = fm.get_lock()
        print "FM lock type:", type(rw2)

        rw_lock.acquireRead()
        rw2.releaseRead()

        print type(rw2.read_lock())

        with rw2.read_lock():
            #: :type: DistanceKernel
            dk = fm.get_distance_kernel()
            print type(dk.row_id_map())

            with SimpleTimer("Getting kernel shape via attribute"):
                _ = dk.get_kernel_matrix().shape

            with SimpleTimer("Getting kernel shape via copy"):
                _ = dk.get_kernel_matrix()[:].shape

    def test_structure_multiprocessing(self):
        print "======================================"
        print "simple multiprocess test of structures"
        print "======================================"

        id_vec = self.mgr.array(self.id_vec)
        bg_ids = self.mgr.array(self.bg_ids)
        feature_mat = self.mgr.matrix(self.feature_mat)
        dk_mat = self.mgr.matrix(self.dk_mat)
        rw_lock = self.mgr.ReadWriteLock()
        #: :type: FeatureMemory
        fm = self.mgr.FeatureMemory(id_vec, bg_ids, feature_mat, dk_mat,
                                    rw_lock)

        rw2 = fm.get_lock()
        print "FM lock type:", type(rw2)

        rw_lock.acquireRead()
        rw2.releaseRead()

        print type(rw2.read_lock())

        with rw2.read_lock():
            #: :type: DistanceKernel
            dk = fm.get_distance_kernel()
            print type(dk.row_id_map())

            with SimpleTimer("Getting kernel shape via attribute"):
                _ = dk.get_kernel_matrix().shape

            with SimpleTimer("Getting kernel shape via copy"):
                _ = dk.get_kernel_matrix()[:].shape

    def test_lock_concurrency(self):
        print "========================================="
        print "multiprocess concurrency test with RWLock"
        print "========================================="

        id_vec = self.mgr.array(self.id_vec)
        bg_ids = self.mgr.array(self.bg_ids)
        feature_mat = self.mgr.matrix(self.feature_mat)
        dk_mat = self.mgr.matrix(self.dk_mat)
        rw_lock = self.mgr.ReadWriteLock()
        #: :type: FeatureMemory
        fm = self.mgr.FeatureMemory(id_vec, bg_ids, feature_mat, dk_mat,
                                    rw_lock)
        dk = fm.get_distance_kernel()

        # noinspection PyShadowingNames
        def test_read_op2(dk, t):
            with dk.get_lock().read_lock():
                print '[tr2] reading'
                time.sleep(t)
                print '[tr2] done'

        # noinspection PyShadowingNames
        def test_write_op2(dk, t):
            with dk.get_lock().write_lock():
                print '[tw2] writing'
                time.sleep(t)
                print '[tw2] done'

        p1 = Process(target=test_read_op2, args=(dk, 10))
        p2 = Process(target=test_write_op2, args=(dk, 3))
        p3 = Process(target=test_read_op2, args=(dk, 5))

        # Baring extreme circumstances, acquires should happen in order of
        # processes being started, i.e. kernel blocks p1 for longer than 0.1
        # seconds, yet allows p2 to go ahead first. Low chance for this to
        # occur.
        s = time.time()
        p1.start()
        time.sleep(0.1)
        p2.start()
        time.sleep(0.1)
        p3.start()

        print "Main PID:", current_process().ident
        print "p1 PID  :", p1.ident
        print "p2 PID  :", p2.ident
        print "p3 PID  :", p3.ident
        # noinspection PyProtectedMember
        print "mgr PID :", self.mgr._process.ident

        # Expected flow: p1 starts first, acquires read lock. p2 starts, but
        # tries to get write lock, which is prevented because of the current
        # read, but is pending. p3 starts, asks for read lock, which is not
        # immediately given because there is a writer pending.
        # p1 finishes after 10 seconds, p2 acquires, p3 blocks.
        # p2 finishes after 3 seconds, p3 acquires.
        # p3 finishes after 5 seconds.
        with SimpleTimer("waiting for p1 to finish "
                         "(should be around 10 seconds)"):
            while p1.is_alive():
                pass
            # taking into account sleeps above
            assert time.time() - s >= 10
            print "Time elapsed since start:", time.time() - s, 's'

        with SimpleTimer("waiting for p2 to finish "
                         "(should be around 3 seconds)"):
            while p2.is_alive():
                pass
            assert time.time() - s >= 13
            print "Time elapsed since start:", time.time() - s, 's'

        with SimpleTimer("waiting for p3 to finish "
                         "(should be around 5 seconds)"):
            while p3.is_alive():
                pass
            assert time.time() - s >= 18
            print "Time elapsed since start:", time.time() - s, 's'

        p1.join()
        p2.join()
        p3.join()

    def test_lock_locality(self):
        # i.e. what process/thread locks are being ID'ed to.
        print "==============================================="
        print "testing lock locality (process/thread location)"
        print "==============================================="

        id_vec = self.mgr.array(self.id_vec)
        bg_ids = self.mgr.array(self.bg_ids)
        feature_mat = self.mgr.matrix(self.feature_mat)
        dk_mat = self.mgr.matrix(self.dk_mat)
        rw_lock = self.mgr.ReadWriteLock()
        #: :type: FeatureMemory
        fm = self.mgr.FeatureMemory(id_vec, bg_ids, feature_mat, dk_mat,
                                    rw_lock)
        dk = fm.get_distance_kernel()

        # these two should be the same lock
        dk_lock = dk.get_lock()
        fm_lock = fm.get_lock()

        print "Main ID we should see:", \
            (current_process().ident, current_thread().ident)
        dk_lock.acquireWrite()
        dk_lock.acquireRead()
        dk_lock.releaseRead()
        dk_lock.releaseWrite()
        print "Should also see the same here:"
        with dk_lock.write_lock():
            with dk_lock.read_lock():
                pass

    def test_feature_extraction_by_id(self):
        print "==============================="
        print "Feature matrix extraction by ID"
        print "==============================="

        id_vec = self.mgr.array(self.id_vec)
        bg_ids = self.mgr.array(self.bg_ids)
        feature_mat = self.mgr.matrix(self.feature_mat)
        dk_mat = self.mgr.matrix(self.dk_mat)
        rw_lock = self.mgr.ReadWriteLock()
        #: :type: FeatureMemory
        fm = self.mgr.FeatureMemory(id_vec, bg_ids, feature_mat, dk_mat,
                                    rw_lock)

        mat = fm.get_feature(1, 5, 80)
        assert mat.shape[0] == 3
        assert (mat[0] == feature_mat[1, :]).all()
        assert (mat[1] == feature_mat[5, :]).all()
        assert (mat[2] == feature_mat[80, :]).all()

    def _update_test_helper(self, fm):
        # new feature info,
        new_id = int(max(fm.get_ids()) + 1)
        feature_len = fm.get_feature(new_id - 1).shape[1]
        new_feature = np.random.random((feature_len,))
        is_bg = True

        #
        # Pre-condition checks
        #
        rw_lock = fm.get_lock()

        # get the distance kernel reference to check later that it mirrors
        # changes made in parent structure.
        print 'getting dist kernel container'
        dk = fm.get_distance_kernel()
        rw_lock.acquireRead()
        assert new_id not in fm.get_ids()
        assert dk.is_symmetric()
        assert new_id not in dk.row_id_map()
        assert new_id not in dk.col_id_map()
        assert new_id not in dk.get_background_ids()
        rw_lock.releaseRead()

        #
        # The update call
        #
        with SimpleTimer("Updating feature memory"):
            fm.update(new_id, new_feature, is_bg)

        #
        # Post-condition checks
        #
        rw_lock.acquireRead()
        # Make sure that we see the new ID in the FM's ID array
        assert new_id in fm.get_ids()
        # If a proxy object is given to the np.allclose method, false is
        # returned. Not sure exactly why. Something to do with the
        # __array_struct__ being transferred from the object server.
        assert np.allclose(fm.get_feature(new_id)[:], new_feature)

        assert dk.is_symmetric()
        assert new_id in dk.row_id_map()
        assert new_id in dk.col_id_map()
        assert new_id in dk.get_background_ids()

        rIDmap, colIDmap, distance_mat = dk.extract_rows(new_id)
        if new_id not in rIDmap:
            raise Exception("new id not correctly in returned matrix ")
        # Should return submat with just background clip distances
        rIDmap, isbgmap, sym_submat = dk.symmetric_submatrix()
        assert new_id in rIDmap
        assert isbgmap[-1]
        rIDmap, isbgmap, sym_submat = dk.symmetric_submatrix(new_id)
        assert new_id in rIDmap
        assert not isbgmap[-1]
        rw_lock.releaseRead()

    def test_feature_updating_local(self):
        print "====================================="
        print "Feature updating with local variables"
        print "====================================="

        print "Type of imported SimpleTimer:", type(SimpleTimer)

        rw_lock = ReadWriteLock()
        #: :type: FeatureMemory
        fm = FeatureMemory(self.id_vec, self.bg_ids, self.feature_mat,
                           self.dk_mat, rw_lock)

        self._update_test_helper(fm)

    def test_feature_updating_remote(self):
        print "====================================="
        print "Feature updating with multiprocessing"
        print "====================================="

        # make proxies of everything.
        print 'ids'
        id_vec = self.mgr.array(self.id_vec)
        print 'bg IDs'
        bg_ids = self.mgr.array(self.bg_ids)
        print "feature matrix"
        feature_mat = self.mgr.matrix(self.feature_mat)
        print "kernel matrix"
        dk_mat = self.mgr.matrix(self.dk_mat)
        print "lock"
        rw_lock = self.mgr.ReadWriteLock()
        #: :type: FeatureMemory
        print "feature manager"
        fm = self.mgr.FeatureMemory(id_vec, bg_ids, feature_mat, dk_mat,
                                    rw_lock)

        self._update_test_helper(fm)

    def test_feature_updating_existing(self):
        print "==============================="
        print "Feature updating - existing CID"
        print "==============================="

        print "( should fail, not implemented yet )"

        fm = FeatureMemory(self.id_vec, self.bg_ids, self.feature_mat,
                           self.dk_mat)

        # new feature info
        new_id = 0
        new_feature = np.random.random((self.M,))
        is_bg = True

        #
        # Pre-condition checks
        #
        rw_lock = fm.get_lock()

        # get the distance kernel reference to check later that it mirrors
        # changes made in parent structure.
        print 'getting dist kernel container'
        dk = fm.get_distance_kernel()
        rw_lock.acquireRead()
        assert new_id in fm.get_ids()
        assert dk.is_symmetric()
        assert new_id in dk.row_id_map()
        assert new_id in dk.col_id_map()
        assert new_id in dk.get_background_ids()

        rw_lock.releaseRead()

        #
        # The update call
        #
        try:
            fm.update(new_id, new_feature, is_bg)
        except NotImplementedError, ex:
            pass
        else:
            assert False, "Should not have passed. Intended functionality " \
                          "not implemented yet."

    def test_map_initialize_type(self):
        type_name_1 = 'cool_feature_thing'
        type_name_2 = 'something_other'
        type_name_3 = 'whoopwhoop'

        cid_file = osp.join(self.data_dir, 'symmetric_clipids.txt')
        bg_flags_file = osp.join(self.data_dir, 'symmetric_bgflags.txt')
        feature_file = osp.join(self.data_dir, 'symmetric_feature.npy')
        kernel_file = osp.join(self.data_dir, 'symmetric_distance_kernel.npy')

        clip_ids = np.array(np.loadtxt(cid_file))
        bg_flags = np.array(np.loadtxt(bg_flags_file))
        # noinspection PyCallingNonCallable
        feature_mat = np.matrix(np.load(feature_file))
        # noinspection PyCallingNonCallable
        kernel_mat = np.matrix(np.load(kernel_file))

        bg_clips = np.array([clip_ids[i]
                             for i, e in enumerate(bg_flags)
                             if e])

        fmm = FeatureMemoryMap()

        # Test forms of initialization
        fmm.initialize(type_name_1, clip_ids, bg_clips, feature_mat, kernel_mat)

        self.assertEqual(len(fmm.get_feature_types()), 1)
        self.assertIn(type_name_1, fmm.get_feature_types())

        self.assertRaises(KeyError, fmm.initialize, type_name_1, clip_ids,
                          bg_clips, feature_mat, kernel_mat)

        fmm.initialize(type_name_2, clip_ids, bg_clips, feature_mat, kernel_mat)
        fmm.initialize_from_files(type_name_3, cid_file, bg_flags_file,
                                  feature_file, kernel_file)

        # Test FMM extraction/pass through methods (also tests uniformity of
        # initialization). All types were set to the same data, so equality
        # checks are valid.
        t1_feats = fmm.get_feature(type_name_1, 3, 15, 37)
        t2_feats = fmm.get_feature(type_name_2, 3, 15, 37)
        t3_feats = fmm.get_feature(type_name_3, 3, 15, 37)

        self.assertTrue(np.allclose(t1_feats, t2_feats) ==
                        np.allclose(t1_feats, t3_feats))

        t1_dk = fmm.get_distance_kernel(type_name_1)
        t2_dk = fmm.get_distance_kernel(type_name_2)
        t3_dk = fmm.get_distance_kernel(type_name_3)

        self.assertTrue(
            np.allclose(t1_dk.row_id_map(), t2_dk.row_id_map()) ==
            np.allclose(t1_dk.row_id_map(), t3_dk.row_id_map())
        )
        self.assertTrue(
            np.allclose(t1_dk.col_id_map(), t2_dk.col_id_map()) ==
            np.allclose(t1_dk.col_id_map(), t3_dk.col_id_map())
        )
        self.assertTrue(
            np.allclose(t1_dk.get_kernel_matrix(), t2_dk.get_kernel_matrix()) ==
            np.allclose(t1_dk.get_kernel_matrix(), t3_dk.get_kernel_matrix())
        )

        # Test removal actually does what its supposed to.
        self.assertIn(type_name_2, fmm.get_feature_types())
        fmm.remove(type_name_2)
        self.assertNotIn(type_name_2, fmm.get_feature_types())
        self.assertRaises(KeyError, fmm.get_feature_memory, type_name_2)

    def test_fmm_singleton(self):
        # test that the correct singleton is being accessed across disjoint
        # processes

        gbl_lock = Lock()
        expected_type = 'expected_type'
        cid_file = osp.join(self.data_dir, 'symmetric_clipids.txt')
        bg_flags_file = osp.join(self.data_dir, 'symmetric_bgflags.txt')
        feature_file = osp.join(self.data_dir, 'symmetric_feature.npy')
        kernel_file = osp.join(self.data_dir, 'symmetric_distance_kernel.npy')

        def test_common_fm(lock):
            lock.acquire()
            lock.release()

            lcl_common_fmm = get_common_fmm()
            assert len(lcl_common_fmm.get_feature_types()) == 0, \
                "local fmm had something in it"
            #: :type: FeatureMemoryMap
            gbl_common_fmm = getFeatureManager().get_common_fmm()
            assert len(gbl_common_fmm.get_feature_types()) >= 1, \
                "Unexpected number of features in global fmm: %s" \
                % gbl_common_fmm.get_feature_types()
            assert expected_type in gbl_common_fmm.get_feature_types(), \
                "Expected feature not present in global fmm. Current " \
                "features: %s" % gbl_common_fmm.get_feature_types()
            gbl_common_fmm.initialize_from_files(
                str(current_process().pid),
                cid_file, bg_flags_file, feature_file, kernel_file
            )

        gbl_lock.acquire()
        print '[main] acquire'

        p1 = Process(name='p1', target=test_common_fm, args=(gbl_lock,))
        p2 = Process(name='p2', target=test_common_fm, args=(gbl_lock,))
        p3 = Process(name='p3', target=test_common_fm, args=(gbl_lock,))

        print '[main] starting processes'
        p1.start()
        p2.start()
        p3.start()

        print '[main] adding expected feature to global FMM'
        #: :type: FeatureMemoryMap
        gbl_fmm = getFeatureManager().get_common_fmm()
        gbl_fmm.initialize_from_files(expected_type, cid_file, bg_flags_file,
                                      feature_file, kernel_file)

        print '[main] release'
        gbl_lock.release()

        p1.join(timeout=1)
        print '[main] checking p1'
        self.assertIsNotNone(p1.exitcode, "p1 timed out")
        self.assertEquals(p1.exitcode, 0, "p1 returned non-0 exitcode")

        p2.join(timeout=1)
        print '[main] checking p2'
        self.assertIsNotNone(p2.exitcode, "p2 timed out")
        self.assertEquals(p2.exitcode, 0, "p2 returned non-0 exitcode")

        p3.join(timeout=1)
        print '[main] checking p3'
        self.assertIsNotNone(p3.exitcode, "p3 timed out")
        self.assertEquals(p3.exitcode, 0, "p3 returned non-0 exitcode")

        print "[main] checking post-processing expected keys"
        self.assertEqual(set(gbl_fmm.get_feature_types()),
                         set((expected_type, str(p1.pid),
                              str(p2.pid), str(p3.pid))))

    def test_fm_external_server(self):
        print "==================================================="
        print "Testing use of external server with update function"
        print "==================================================="

        host = 'localhost'
        port = 54321
        addr = (host, port)
        auth_key = "foobar"

        # Trying to connect to these right away should result in a timeout
        # in the underlying proxy.
        self.assertRaises(
            Exception,
            initFeatureManagerConnection,
            addr, auth_key
        )

        # Start the server at the specified addr
        config_file = osp.join(self.work_dir, 'test_server_config.ini')
        config = SafeConfigCommentParser()
        config.add_section('server')
        config.set('server', 'port', str(port))
        config.set('server', 'authkey', auth_key)
        with open(config_file, 'w') as ofile:
            # noinspection PyTypeChecker
            config.write(ofile)
        p = subprocess.Popen(['FeatureManagerServer', '-c', config_file])
        # wait a sec for the server to start/fail
        time.sleep(1)  # should be sufficient...
        self.assertIsNone(p.poll(), "Server process shutdown prematurely. "
                                    "Check reported error on console.")

        # Now we should be able to connect, create a FeatureMemory[Map] and
        # successfully go through the update process.
        from multiprocessing.connection import AuthenticationError
        self.assertRaises(
            AuthenticationError,
            initFeatureManagerConnection,
            addr, "not the right key"
        )

        # Shouldn't be in there yet as its not initialized
        self.assertRaises(
            KeyError,
            getFeatureManager,
            addr
        )

        initFeatureManagerConnection(addr, auth_key)

        # should get a ValueError when trying to init the same address more than
        # once.
        self.assertRaises(
            ValueError,
            initFeatureManagerConnection,
            addr, auth_key
        )

        mgr = getFeatureManager(addr)

        # Create a feature map->featureMemory and pass through update process
        f_type = 'test'
        cid_file = osp.join(self.data_dir, 'symmetric_clipids.txt')
        bg_flags_file = osp.join(self.data_dir, 'symmetric_bgflags.txt')
        feature_file = osp.join(self.data_dir, 'symmetric_feature.npy')
        kernel_file = osp.join(self.data_dir, 'symmetric_distance_kernel.npy')

        #: :type: FeatureMemoryMap
        fmm = mgr.get_common_fmm()
        fmm.initialize_from_files(f_type, cid_file, bg_flags_file, feature_file,
                                  kernel_file)
        fm = fmm.get_feature_memory(f_type)
        self._update_test_helper(fm)

        # Clean up
        # Need to call del else we would get hung up in decref call to remove
        # server at function/process exit.
        del fm, fmm
        p.terminate()
        os.remove(config_file)
        removeFeatureManagerConnection(addr)

    def test_symm_dk_create_from_file(self):
        print "====================================================="
        print "Testing remote file-based DistanceKernel construction"
        print "====================================================="

        s_cid_file = osp.join(self.data_dir, 'symmetric_clipids.txt')
        s_bg_flags_file = osp.join(self.data_dir, 'symmetric_bgflags.txt')
        s_ker_file = osp.join(self.data_dir, 'symmetric_distance_kernel.npy')
        a_row_cids = osp.join(self.data_dir, 'asymmetric_clipids_rows.txt')
        a_col_cids = osp.join(self.data_dir, 'asymmetric_clipids_cols.txt')
        a_ker_file = osp.join(self.data_dir, 'asymmetric_distance_kernel.npy')

        symm_dk = self.mgr.symmetric_dk_from_file(s_cid_file,
                                                  s_ker_file)
        self.assertTrue(symm_dk.is_symmetric())
        self.assertEquals(symm_dk.get_background_ids().shape, (0,))
        self.assertEqual(symm_dk.symmetric_submatrix()[2].shape, (0, 0))

        symm_dk_wBGs = self.mgr.symmetric_dk_from_file(s_cid_file,
                                                       s_ker_file,
                                                       s_bg_flags_file)
        self.assertTrue(symm_dk_wBGs.is_symmetric())
        self.assertEqual(symm_dk_wBGs.get_background_ids().shape, (40,))
        self.assertEqual(symm_dk_wBGs.symmetric_submatrix()[2].shape, (40, 40))

        asym_dk = self.mgr.asymmetric_dk_from_file(a_row_cids,
                                                   a_col_cids,
                                                   a_ker_file)
        self.assertFalse(asym_dk.is_symmetric())
        self.assertEqual(asym_dk.get_background_ids().shape, (0,))
        self.assertRaises(RuntimeError, asym_dk.symmetric_submatrix, ())
        self.assertEqual(asym_dk.extract_rows(0, 17, 43)[2].shape, (3, 10000))

    def test_controller_interaction(self):
        print 'poke'
        # Dummy classes representing function of intended components
        class Controller (Process):
            """ Reads input from queue, exiting when receives None
            """
            def __init__(self, data_queue):
                super(Controller, self).__init__(name="controller")
                self._queue = data_queue
            def run(self):
                running = True
                while running:
                    print "[Controller] waiting for data..."
                    elem = self._queue.get()
                    print "[Controller] received:", type(elem), elem
                    if elem is None:
                        running = False
        class DummyAsw (Process):
            """ sends dk to queue on separate process
            """
            def __init__(self, dk, data_queue):
                super(DummyAsw, self).__init__()
                self._dk = dk
                self._queue = data_queue
            def run(self):
                self._queue.put(self._dk)
        def generate_dks(mgr):
            """ generated some distance kernels to test with given a connected
                FeatureManager
            """
            c_fmm = mgr.get_common_fmm()
            if 'test' in c_fmm.get_feature_types():
                c_fmm.remove('test')
            c_fmm.initialize_from_files('test', self.sym_cids_file,
                                        self.sym_bg_flags,
                                        self.sym_feat_mat_f,
                                        self.sym_dk_mat_f)
            dk1 = c_fmm.get_distance_kernel('test')
            dk2 = mgr.symmetric_dk_from_file(self.sym_cids_file,
                                             self.sym_dk_mat_f,
                                             self.sym_bg_flags)
            return dk1, dk2

        # Set-up and start the extra-process feature manager server
        host = 'localhost'
        port = 54321
        addr = (host, port)
        auth_key = "foobar"
        config_file = osp.join(self.work_dir,
                               'controller_interaction_server.ini')
        config = SafeConfigCommentParser()
        config.add_section('server')
        config.set('server', 'port', str(port))
        config.set('server', 'authkey', auth_key)
        with open(config_file, 'w') as ofile:
            # noinspection PyTypeChecker
            config.write(ofile)
        p = subprocess.Popen(['FeatureManagerServer', '-c', config_file])
        # wait a sec for the server to start/fail
        time.sleep(1)  # should be sufficient...
        self.assertIsNone(p.poll(), "Server process shutdown prematurely. "
                                    "Check reported error on console.")

        initFeatureManagerConnection(addr, auth_key)
        mgr = getFeatureManager(addr)

        # Initialize and start dummy controller process
        q = Queue()
        c = Controller(q)
        c.start()

        # generated DistanceKernels and dummy ASW processes
        dk1, dk2 = generate_dks(mgr)
        asw1 = DummyAsw(dk1, q)
        asw2 = DummyAsw(dk2, q)

        # Running the two dummy asw processes should cause no errors in either
        # dummy ASW or the dummy Controller
        asw1.start()
        asw1.join()
        asw2.start()
        asw2.join()

        # shutdown controller
        q.put(None)
        c.join()
        print "C exitcode:", c.exitcode

        # Clean-up
        # Need to call del on some things else we would get hung up in decref
        # call to remove server at function/process exit.
        del dk1, dk2, asw1, asw2
        os.remove(config_file)
        removeFeatureManagerConnection(addr)
        p.terminate()
        p.poll()

        self.assertEqual(c.exitcode, 0, "Controller dummy did not exit cleanly")
        del c
