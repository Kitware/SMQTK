import mock
import nose.tools as ntools
import numpy
import unittest

from smqtk.representation import DescriptorElement
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement


__author__ = 'purg'


class DummyElementImpl (DescriptorElement):

    def __init__(self, *args, **kwds):
        pass

    def set_vector(self, new_vec):
        pass

    def has_vector(self):
        pass

    def vector(self):
        pass

    def get_config(self):
        pass


class TestDescriptorElemFactory (unittest.TestCase):

    @mock.patch.object(DummyElementImpl, "__init__")
    def test_no_params(self, dei_init):
        # So we don't break python
        dei_init.return_value = None

        test_params = {}

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        expected_type = 'type'
        expected_uuid = 'uuid'
        # Should construct a new DEI instance under they hood somewhere
        r = factory.new_descriptor(expected_type, expected_uuid)

        ntools.assert_true(dei_init.called)
        dei_init.assert_called_once_with(expected_type, expected_uuid)
        ntools.assert_is_instance(r, DummyElementImpl)

    @mock.patch.object(DummyElementImpl, "__init__")
    def test_with_params(self, dei_init):
        # So we don't break python
        dei_init.return_value = None

        v = numpy.random.randint(0, 10, 10)
        test_params = {
            'p1': 'some dir',
            'vec': v
        }

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        ex_type = 'type'
        ex_uuid = 'uuid'
        # Should construct a new DEI instance under they hood somewhere
        r = factory.new_descriptor(ex_type, ex_uuid)

        ntools.assert_true(dei_init.called)
        dei_init.assert_called_once_with(ex_type, ex_uuid, p1='some dir', vec=v)
        ntools.assert_is_instance(r, DummyElementImpl)

    @mock.patch.object(DummyElementImpl, "__init__")
    def test_call(self, dei_init):
        # So we don't break python
        dei_init.return_value = None

        # Same as `test_with_params` but using __call__ entry point
        v = numpy.random.randint(0, 10, 10)
        test_params = {
            'p1': 'some dir',
            'vec': v
        }

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        ex_type = 'type'
        ex_uuid = 'uuid'
        # Should construct a new DEI instance under they hood somewhere
        r = factory(ex_type, ex_uuid)

        ntools.assert_true(dei_init.called)
        dei_init.assert_called_once_with(ex_type, ex_uuid, p1='some dir', vec=v)
        ntools.assert_is_instance(r, DummyElementImpl)

    def test_configuration(self):
        c = DescriptorElementFactory.get_default_config()
        ntools.assert_is_none(c['type'])
        ntools.assert_in('DescriptorMemoryElement', c)

        c['type'] = 'DescriptorMemoryElement'
        factory = DescriptorElementFactory.from_config(c)
        ntools.assert_equal(factory._d_type.__name__,
                            DescriptorMemoryElement.__name__)
        ntools.assert_equal(factory._d_type_config, {})

        d = factory.new_descriptor('test', 'foo')
        ntools.assert_equal(d.type(), 'test')
        ntools.assert_equal(d.uuid(), 'foo')
