import unittest

import nose.tools as ntools

from smqtk.iqr_index.libsvm_hik import LibSvmHikIqrIndex


__author__ = 'purg'


class TestIqrSvmHik (unittest.TestCase):

    def test_configuration(self):
        c = LibSvmHikIqrIndex.default_config()
        ntools.assert_in('descr_cache_filepath', c)

        # change default for something different
        c['descr_cache_filepath'] = 'foobar.thing'

        iqr_index = LibSvmHikIqrIndex.from_config(c)
        ntools.assert_equal(iqr_index._descr_cache_fp,
                            c['descr_cache_filepath'])

        # test config idempotency
        ntools.assert_dict_equal(c, iqr_index.get_config())
