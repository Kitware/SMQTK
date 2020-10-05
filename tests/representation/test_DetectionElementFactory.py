import unittest.mock as mock

from smqtk.representation import DetectionElement
from smqtk.representation.detection_element_factory \
    import DetectionElementFactory


def test_get_default_config():
    """
    Test that get_default_config method does not error and returns a
    dictionary with a ``type`` key.
    """
    c = DetectionElementFactory.get_default_config()
    assert isinstance(c, dict)
    assert 'type' in c


@mock.patch.object(DetectionElementFactory, '__init__')
@mock.patch.object(DetectionElementFactory, 'get_default_config')
@mock.patch.object(DetectionElement, 'get_impls')
def test_from_config_no_merge(m_de_get_impls, m_def_get_default_config,
                              m_def_init):
    """
    Test that ``from_config`` appropriately constructs a factory instance
    without merging a default configuration.
    """
    # Because __init__ needs to return None
    m_def_init.return_value = None

    # Mock available implementations of DetectionElement
    T1 = mock.MagicMock(spec=DetectionElement)
    T1.__name__ = 'T1'
    T1.__module__ = __name__
    T2 = mock.MagicMock(spec=DetectionElement)
    T2.__name__ = 'T2'
    T2.__module__ = __name__
    expected_impls_set = {T1, T2}
    m_de_get_impls.return_value = expected_impls_set

    # Mock default configuration return from class method.
    expected_default_conf = {
        'type': None,
        f'{__name__}.T1': {'z': 'z'},
        f'{__name__}.T2': {'y': 'y'},
    }
    m_def_get_default_config.return_value = expected_default_conf

    # Test configuration we are passing to ``from_config``.
    test_config = {'type': f'{__name__}.T2',
                   f'{__name__}.T2': {'a': 1, 'b': 'c'}}

    # Because we are not merging default config, we expect only the contents
    # of the passed config to reach the factory constructor.
    expected_type = T2
    expected_conf = {'a': 1, 'b': 'c'}

    DetectionElementFactory.from_config(test_config, merge_default=False)

    m_def_get_default_config.assert_not_called()
    m_de_get_impls.assert_called_once()
    m_def_init.assert_called_once_with(expected_type, expected_conf)


@mock.patch.object(DetectionElementFactory, '__init__')
@mock.patch.object(DetectionElementFactory, 'get_default_config')
@mock.patch.object(DetectionElement, 'get_impls')
def test_from_config_with_merge(m_de_get_impls, m_def_get_default_config,
                                m_def_init):
    """
    Test that ``from_config`` appropriately constructs a factory instance
    after merging the default configuration.
    """
    # Because __init__ needs to return None
    m_def_init.return_value = None

    # Mock available implementations of DetectionElement
    # - Overriding class location to be "local" for testing.
    T1 = mock.MagicMock(spec=DetectionElement)
    T1.__name__ = 'T1'
    T1.__module__ = __name__
    T2 = mock.MagicMock(spec=DetectionElement)
    T2.__name__ = 'T2'
    T2.__module__ = __name__
    expected_impls_set = {T1, T2}
    m_de_get_impls.return_value = expected_impls_set

    # Mock default configuration return from class method.
    expected_default_conf = {
        'type': None,
        f'{__name__}.T1': {'z': 'z'},
        f'{__name__}.T2': {'y': 'y'},
    }
    m_def_get_default_config.return_value = expected_default_conf

    # Partial configuration to pass to ``from_config``.
    test_config = {'type': f'{__name__}.T2',
                   f'{__name__}.T2': {'a': 1, 'b': 'c'}}

    # Expected construction values. Note that conf has default component(s)
    # merged into it.
    expected_type = T2
    expected_conf = {'a': 1, 'b': 'c', 'y': 'y'}

    DetectionElementFactory.from_config(test_config, merge_default=True)

    m_def_get_default_config.assert_called_once()
    m_de_get_impls.assert_called_once()
    m_def_init.assert_called_once_with(expected_type, expected_conf)


def test_get_config():
    """
    Test that ``get_config`` returns the appropriate configuration dictionary.
    """
    test_type = mock.MagicMock(spec=DetectionElement)
    test_type.__name__ = 'T1'
    test_type.__module__ = __name__
    test_conf = {'a': 1, 'b': 'c'}

    expected_config = {
        "type": f"{__name__}.T1",
        f"{__name__}.T1": {'a': 1, 'b': 'c'}
    }

    # noinspection PyTypeChecker
    factory = DetectionElementFactory(test_type, test_conf)
    assert factory.get_config() == expected_config


def test_new_detection_function():
    """
    Test that the given type and config at construction time is used to
    create a new instance via known ``Configurable`` interface methods.
    """
    elem_type = mock.MagicMock(spec=DetectionElement)
    # store expected function return that should be returned from
    # ``new_detection`` call.
    expected_from_config_return = elem_type.from_config.return_value

    elem_config = {'a': 1, 'b': 'c'}
    expected_uuid = 'some uuid'

    # noinspection PyTypeChecker
    test_factory = DetectionElementFactory(elem_type, elem_config)
    assert test_factory.new_detection(expected_uuid) == \
        expected_from_config_return

    elem_type.from_config.assert_called_once_with(elem_config, expected_uuid)


def test_new_detection_call_hook():
    """
    Same as ``test_new_detection_function`` but invoking through __call__ hook.
    """
    elem_type = mock.MagicMock(spec=DetectionElement)
    # store expected function return that should be returned from
    # ``new_detection`` call.
    expected_from_config_return = elem_type.from_config.return_value

    elem_config = {'a': 1, 'b': 'c'}
    expected_uuid = 'some uuid'

    # noinspection PyTypeChecker
    test_factory = DetectionElementFactory(elem_type, elem_config)
    # Point of distinction: use of __call__
    # noinspection PyArgumentList
    assert test_factory(expected_uuid) == expected_from_config_return

    elem_type.from_config.assert_called_once_with(elem_config, expected_uuid)
