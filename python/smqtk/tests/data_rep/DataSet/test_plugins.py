
from smqtk.data_rep.data_set_impl import get_data_set_impls


__author__ = 'purg'


def test_plugin_getter():
    c = get_data_set_impls()
    assert 'DataFileSet' in c
