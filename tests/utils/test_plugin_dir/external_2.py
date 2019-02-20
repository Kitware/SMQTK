from tests.utils.test_plugin_dir.internal_plugins.interface import \
    DummyInterface


class ImplExternal3 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'external3'+str(val)
