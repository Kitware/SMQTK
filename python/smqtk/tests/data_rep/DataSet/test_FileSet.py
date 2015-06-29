__author__ = 'purg'

import mock
import nose.tools as ntools
import unittest

from smqtk.data_rep.data_element_impl.memory_element import DataMemoryElement
from smqtk.data_rep.data_set_impl.file_set import DataFileSet


class TestDataFileSet (unittest.TestCase):

    @mock.patch.object(DataFileSet, '_save_data_elements')
    def test_data_serialization(self, m_dfs_sde):
        save_dir = '/not/a/real/dir'

        ds = DataFileSet(save_dir)
        del ds
        ntools.assert_false(m_dfs_sde.called)

        ds = DataFileSet(save_dir)
        de = DataMemoryElement('foo', 'text/plain')
        ds.add_data(de)
        del ds
        ntools.assert_true(m_dfs_sde.called)
