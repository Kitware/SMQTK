import mock
import nose.tools as ntools
import numpy
import unittest

from smqtk.representation import DescriptorElement
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement


__author__ = "paul.tunison@kitware.com"


class DummyElementImpl (DescriptorElement):

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_str, uuid, *args, **kwds):
        super(DummyElementImpl, self).__init__(type_str, uuid)
        self.args = args
        self.kwds = kwds

    def set_vector(self, new_vec):
        pass

    def has_vector(self):
        pass

    def vector(self):
        pass

    def get_config(self):
        pass


class TestDescriptorElemFactory (unittest.TestCase):

    def test_no_params(self):
        test_params = {}

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        expected_type = 'type'
        expected_uuid = 'uuid'
        expected_args = ()
        expected_kwds = {}

        # Should construct a new DEI instance under they hood somewhere
        r = factory.new_descriptor(expected_type, expected_uuid)

        ntools.assert_is_instance(r, DummyElementImpl)
        ntools.assert_equal(r._type_label, expected_type)
        ntools.assert_equal(r._uuid, expected_uuid)
        ntools.assert_equal(r.args, expected_args)
        ntools.assert_equal(r.kwds, expected_kwds)

    def test_with_params(self):
        v = numpy.random.randint(0, 10, 10)
        test_params = {
            'p1': 'some dir',
            'vec': v
        }

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        ex_type = 'type'
        ex_uuid = 'uuid'
        ex_args = ()
        ex_kwds = test_params
        # Should construct a new DEI instance under they hood somewhere
        r = factory.new_descriptor(ex_type, ex_uuid)

        ntools.assert_is_instance(r, DummyElementImpl)
        ntools.assert_equal(r._type_label, ex_type)
        ntools.assert_equal(r._uuid, ex_uuid)
        ntools.assert_equal(r.args, ex_args)
        ntools.assert_equal(r.kwds, ex_kwds)

    def test_call(self):
        # Same as `test_with_params` but using __call__ entry point
        v = numpy.random.randint(0, 10, 10)
        test_params = {
            'p1': 'some dir',
            'vec': v
        }

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        ex_type = 'type'
        ex_uuid = 'uuid'
        ex_args = ()
        ex_kwds = test_params
        # Should construct a new DEI instance under they hood somewhere
        r = factory(ex_type, ex_uuid)

        ntools.assert_is_instance(r, DummyElementImpl)
        ntools.assert_equal(r._type_label, ex_type)
        ntools.assert_equal(r._uuid, ex_uuid)
        ntools.assert_equal(r.args, ex_args)
        ntools.assert_equal(r.kwds, ex_kwds)

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
