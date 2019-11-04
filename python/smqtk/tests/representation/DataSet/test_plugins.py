from smqtk.representation.data_set import DataSet


def test_plugin_getter():
    c = DataSet.get_impls()
    assert isinstance(c, dict)
    # The following implementations at least should always be available.
    assert 'DataFileSet' in c
    assert 'DataMemorySet' in c
