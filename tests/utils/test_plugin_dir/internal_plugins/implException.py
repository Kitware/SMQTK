"""
Example of a module that has an exception upon import.
"""
from tests.utils.test_plugin_dir.internal_plugins.interface import \
    DummyInterface

raise RuntimeError('some intentional error to trigger on import')


class ImplActuallyValid (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return "actuallyValidImpl"+str(val)
