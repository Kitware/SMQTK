import os
import tempfile
import unittest

import mock
import nose.tools
import numpy

from smqtk.algorithms.nn_index.hash_index.sklearn_balltree import \
    SkLearnBallTreeHashIndex


__author__ = "paul.tunison@kitware.com"


if SkLearnBallTreeHashIndex.is_usable():

    class TestBallTreeHashIndex (unittest.TestCase):

        def test_file_cache_type(self):
            # Requites .npz file
            nose.tools.assert_raises(
                ValueError,
                SkLearnBallTreeHashIndex,
                file_cache='some_file.txt'
            )

            SkLearnBallTreeHashIndex(file_cache='some_file.npz')

        @mock.patch('smqtk.algorithms.nn_index.hash_index.sklearn_balltree.numpy.savez')
        def test_save_model_no_cache(self, m_savez):
            bt = SkLearnBallTreeHashIndex()
            m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
            bt.build_index(m)
            nose.tools.assert_false(m_savez.called)

        @mock.patch('smqtk.algorithms.nn_index.hash_index.sklearn_balltree.numpy.savez')
        def test_save_model_with_cache(self, m_savez):
            bt = SkLearnBallTreeHashIndex('some_file.npz')
            m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
            bt.build_index(m)
            nose.tools.assert_true(m_savez.called)
            nose.tools.assert_equal(m_savez.call_count, 1)

        def test_model_reload(self):
            fd, fp = tempfile.mkstemp('.npz')
            os.close(fd)
            os.remove(fp)  # shouldn't exist before construction
            try:
                bt = SkLearnBallTreeHashIndex(fp)
                m = numpy.random.randint(0, 2, 1000 * 256).reshape(1000, 256)
                bt.build_index(m)
                q = numpy.random.randint(0, 2, 256).astype(bool)
                bt_neighbors, bt_dists = bt.nn(q, 10)

                bt2 = SkLearnBallTreeHashIndex(fp)
                bt2_neighbors, bt2_dists = bt2.nn(q, 10)

                nose.tools.assert_is_not(bt, bt2)
                nose.tools.assert_is_not(bt.bt, bt2.bt)
                numpy.testing.assert_equal(bt2_neighbors, bt_neighbors)
                numpy.testing.assert_equal(bt2_dists, bt_dists)
            finally:
                os.remove(fp)

        def test_invalid_build(self):
            bt = SkLearnBallTreeHashIndex()
            nose.tools.assert_raises(
                ValueError,
                bt.build_index,
                []
            )

        def test_get_config(self):
            bt = SkLearnBallTreeHashIndex()
            bt_c = bt.get_config()

            nose.tools.assert_equal(len(bt_c), 3)
            nose.tools.assert_in('file_cache', bt_c)
            nose.tools.assert_in('leaf_size', bt_c)
            nose.tools.assert_in('random_seed', bt_c)

            nose.tools.assert_equal(bt_c['file_cache'], None)
