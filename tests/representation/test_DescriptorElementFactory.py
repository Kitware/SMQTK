import unittest

import numpy

from smqtk.representation import DescriptorElement
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement


class DummyElementImpl (DescriptorElement):

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_str, uuid, *args, **kwds):
        super(DummyElementImpl, self).__init__(type_str, uuid)
        self.args = args
        self.kwds = kwds

    def set_vector(self, new_vec):
        return self

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

        self.assertIsInstance(r, DummyElementImpl)
        self.assertEqual(r._type_label, expected_type)
        self.assertEqual(r._uuid, expected_uuid)
        self.assertEqual(r.args, expected_args)
        self.assertEqual(r.kwds, expected_kwds)

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

        self.assertIsInstance(r, DummyElementImpl)
        self.assertEqual(r._type_label, ex_type)
        self.assertEqual(r._uuid, ex_uuid)
        self.assertEqual(r.args, ex_args)
        self.assertEqual(r.kwds, ex_kwds)

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

        self.assertIsInstance(r, DummyElementImpl)
        self.assertEqual(r._type_label, ex_type)
        self.assertEqual(r._uuid, ex_uuid)
        self.assertEqual(r.args, ex_args)
        self.assertEqual(r.kwds, ex_kwds)

    def test_configuration(self):
        c = DescriptorElementFactory.get_default_config()
        self.assertIsNone(c['type'])
        dme_key = 'smqtk.representation.descriptor_element.local_elements.DescriptorMemoryElement'
        self.assertIn(dme_key, c)

        c['type'] = dme_key
        factory = DescriptorElementFactory.from_config(c)
        self.assertEqual(factory._d_type.__name__,
                         DescriptorMemoryElement.__name__)
        self.assertEqual(factory._d_type_config, {})

        d = factory.new_descriptor('test', 'foo')
        self.assertEqual(d.type(), 'test')
        self.assertEqual(d.uuid(), 'foo')

    def test_get_config(self):
        """
        We should be able to get the configuration of the current factory.
        This should look like the same as the
        """
        test_params = {
            'p1': 'some dir',
            'vec': 1
        }
        dummy_key = f"{__name__}.{DummyElementImpl.__name__}"
        factory = DescriptorElementFactory(DummyElementImpl, test_params)
        factory_config = factory.get_config()
        assert factory_config == {"type": dummy_key,
                                  dummy_key: test_params}
