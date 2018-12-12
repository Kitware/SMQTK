"""
Some valid plugins to load via extension successfully.
"""
from tests.utils.test_plugin_dir.internal_plugins.interface \
    import DummyInterface


class ValidExtensionPlugin (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return "ValidExtensionPlugin" + val


class UnusableExtensionPlugin (DummyInterface):

    @classmethod
    def is_usable(cls):
        return False

    def inst_method(self, val):
        return "ValidExtensionPlugin" + val
