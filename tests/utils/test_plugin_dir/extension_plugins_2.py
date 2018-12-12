"""
Some valid plugins to load via extension successfully.
"""
from tests.utils.test_plugin_dir.internal_plugins.interface \
    import DummyInterface


class ValidExtensionPlugin2 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return "ValidExtensionPlugin" + val
