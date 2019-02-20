"""
Example of a module that has an exception upon import.
"""
from smqtk.tests.utils.test_plugin_dir.internal_plugins.interface import \
    DummyInterface


class ImplActuallyValid (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return "actuallyValidImpl"+str(val)


raise RuntimeError('some intentional error to trigger on import')
