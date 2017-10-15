import os
import tempfile
import unittest

import mock
import nose.tools
import numpy

from smqtk.algorithms.nn_index.hash_index.vptree import \
    VPTreeHashIndex


__author__ = "william.p.hicks@gmail.com"


if VPTreeHashIndex.is_usable():

    class TestVPTreeHashIndex (unittest.TestCase):

        def test_file_cache_type(self):
            # Requires .npz file
            nose.tools.assert_raises(
                ValueError,
                VPTreeHashIndex,
                file_cache='some_file.txt'
            )

            VPTreeHashIndex(file_cache='some_file.npz')

        @mock.patch('smqtk.algorithms.nn_index.hash_index.vptree.numpy.savez')
        def test_save_model_no_cache(self, m_savez):
            vpt = VPTreeHashIndex()
            m = numpy.random.randint(0, 2, size=(1000, 256))
            vpt.build_index(m)
            nose.tools.assert_false(m_savez.called)

        @mock.patch('smqtk.algorithms.nn_index.hash_index.vptree.numpy.savez')
        def test_save_model_with_cache(self, m_savez):
            vpt = VPTreeHashIndex('some_file.npz')
            m = numpy.random.randint(0, 2, size=(1000, 256), dtype='bool')
            vpt.build_index(m)
            nose.tools.assert_true(m_savez.called)
            nose.tools.assert_equal(m_savez.call_count, 1)

        def test_model_reload(self):
            fd, fp = tempfile.mkstemp('.npz')
            os.close(fd)
            os.remove(fp)  # shouldn't exist before construction
            try:
                vpt = VPTreeHashIndex(file_cache=fp)
                m = numpy.random.randint(0, 2, size=(1000, 256), dtype='bool')
                vpt.build_index(m)
                q = numpy.random.randint(0, 2, 256, dtype='bool')
                vpt_neighbors, vpt_dists = vpt.nn(q, 10)

                vpt2 = VPTreeHashIndex(file_cache=fp)
                vpt2_neighbors, vpt2_dists = vpt2.nn(q, 10)

                nose.tools.assert_is_not(vpt, vpt2)
                nose.tools.assert_is_not(vpt.vpt, vpt2.vpt)
                numpy.testing.assert_equal(vpt2_neighbors, vpt_neighbors)
                numpy.testing.assert_equal(vpt2_dists, vpt_dists)
            finally:
                os.remove(fp)

        def test_invalid_build(self):
            vpt = VPTreeHashIndex()
            nose.tools.assert_raises(
                ValueError,
                vpt.build_index,
                []
            )

        def test_get_config(self):
            vpt = VPTreeHashIndex()
            vpt_c = vpt.get_config()

            nose.tools.assert_equal(len(vpt_c), 3)
            nose.tools.assert_in('file_cache', vpt_c)
            nose.tools.assert_in('random_seed', vpt_c)
            nose.tools.assert_in('tree_type', vpt_c)

            nose.tools.assert_is(vpt_c['file_cache'], None)
