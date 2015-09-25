
from smqtk.representation.data_set import get_data_set_impls


__author__ = "paul.tunison@kitware.com"


def test_plugin_getter():
    c = get_data_set_impls()
    assert 'DataFileSet' in c
