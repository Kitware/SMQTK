from __future__ import division, print_function
from smqtk.representation.data_set import get_data_set_impls


def test_plugin_getter():
    c = get_data_set_impls()
    assert isinstance(c, dict)
    # The following implementations at least should always be available.
    assert 'DataFileSet' in c
    assert 'DataMemorySet' in c
