from tests.utils.test_plugin_dir.internal_plugins.interface import \
    DummyInterface


# Intentionally still abstract for testing.
# noinspection PyAbstractClass
class ImplExternal7 (DummyInterface):

    @classmethod
    def is_usable(cls):
        return True

    # Intentionally missing implementation of ``inst_method``
