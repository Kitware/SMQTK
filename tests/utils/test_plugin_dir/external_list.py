from tests.utils.test_plugin_dir.internal_plugins.interface import \
    DummyInterface


class ImplExternal4 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'external4'+str(val)


class ImplExternal5 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'external5'+str(val)


class ImplExternal6 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'external6'+str(val)


TEST_PLUGIN_CLASS = [ImplExternal4, ImplExternal5]
