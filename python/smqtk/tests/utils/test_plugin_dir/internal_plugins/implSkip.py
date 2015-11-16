from smqtk.tests.utils.test_plugin_get import DummyInterface


__author__ = 'paul.tunison@kitware.com'


class ImplSkipModule (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    def inst_method(self, val):
        return "skipModule"+str(val)


TEST_PLUGIN_CLASS = None
