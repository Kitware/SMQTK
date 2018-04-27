from smqtk.tests.utils.test_plugin_dummy_interface import DummyInterface


class ImplExternal7 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True
