"""
Example where only explicitly listed classes are exported even though others are
implemented.
"""

from smqtk.tests.utils.test_plugin_get import DummyInterface


class ImplDoExport (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'doExport'+str(val)


class ImplNoExport (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'noExport'+str(val)


TEST_PLUGIN_CLASS = ImplDoExport
