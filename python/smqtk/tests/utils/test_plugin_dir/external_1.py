from smqtk.tests.utils.test_plugin_get import DummyInterface


__author__ = 'paul.tunison@kitware.com'


class ImplExternal1 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'external1'+str(val)


class ImplExternal2 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return 'external2'+str(val)
