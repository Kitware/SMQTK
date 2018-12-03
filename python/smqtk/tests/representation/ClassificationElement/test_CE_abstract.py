from __future__ import division, print_function
import unittest

import mock
import pytest

import smqtk.representation
import smqtk.exceptions


class DummyCEImpl (smqtk.representation.ClassificationElement):

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def get_classification(self):
        pass

    def set_classification(self, m=None, **kwds):
        return super(DummyCEImpl, self).set_classification(m, **kwds)

    def has_classifications(self):
        pass


class TestClassificationElementAbstract (unittest.TestCase):

    def test_init(self):
        e = DummyCEImpl('foo', 'bar')
        self.assertEqual(e.type_name, 'foo')
        self.assertEqual(e.uuid, 'bar')

    def test_hash(self):
        self.assertEqual(hash(DummyCEImpl('foo', 'bar')),
                         hash(('foo', 'bar')))

    def test_equality(self):
        e1 = DummyCEImpl('test', 0)
        e2 = DummyCEImpl('other', 1)
        e1.get_classification = e2.get_classification = \
            mock.Mock(return_value={1: 1, 2: 0})

        self.assertEqual(e1, e2)

    def test_not_equal(self):
        e1 = DummyCEImpl('test', 0)
        e1.get_classification = mock.Mock(return_value={1: 1, 2: 0})

        e2 = DummyCEImpl('other', 1)
        e2.get_classification = mock.Mock(return_value={1: 0, 2: 1})

        self.assertNotEqual(e1, e2)

    def test_get_items(self):
        e1 = DummyCEImpl('test', 0)
        e1.get_classification = mock.Mock(return_value={1: 1, 2: 0})

        # There should not be a KeyError for accessing the test labels, but
        # there should for a different label that is not includes in the element
        self.assertEqual(e1[1], 1)
        self.assertEqual(e1[2], 0)
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            e1['some other key']

    def test_max_label(self):
        e = DummyCEImpl('test', 0)

        e.get_classification = mock.Mock(return_value={})
        self.assertRaises(
            smqtk.exceptions.NoClassificationError,
            e.max_label
        )

        e.get_classification = mock.Mock(return_value={1: 0, 2: 1})
        self.assertNotEqual(e.max_label(), 1)
        self.assertEqual(e.max_label(), 2)

    def test_set_no_input(self):
        e = DummyCEImpl('test', 0)
        self.assertRaises(
            ValueError,
            e.set_classification,
        )

    def test_set_empty_input(self):
        e = DummyCEImpl('test', 0)
        self.assertRaises(
            ValueError,
            e.set_classification,
            {}
        )

    def test_set_input_dict(self):
        e = DummyCEImpl('test', 0)
        v = {1: 0, 2: 1}
        self.assertEqual(e.set_classification(v),  v)

    def test_set_kwargs(self):
        e = DummyCEImpl('test', 0)
        self.assertEqual(e.set_classification(a=1, b=0),
                         {'a': 1, 'b': 0})

    def test_set_mixed(self):
        e = DummyCEImpl('test', 0)
        self.assertEqual(
            e.set_classification({'a': .25, 1: .25},
                                 b=.25, d=.25),
            {'a': .25, 1: .25, 'b': .25, 'd': .25}
        )

    def test_set_nonstandard(self):
        # Many classifiers output 1-sum confidence values, but not all (e.g.
        # CNN final layers like AlexNet).
        e = DummyCEImpl('test', 0)
        self.assertEqual(
            e.set_classification({'a': 1, 1: 1},
                                 b=1, d=1),
            {'a': 1, 1: 1, 'b': 1, 'd': 1}
        )


class TestClassificationElementAbstractImplGetter (unittest.TestCase):

    def test_get_plugins(self):
        # There are at least 2 internally provided implementations that will
        # always be available:
        #   MemoryClassificationElement
        #   FileClassificationElement
        m = smqtk.representation.get_classification_element_impls()
        assert len(m) >= 2
        self.assertIn('MemoryClassificationElement', m)
        self.assertIn('FileClassificationElement', m)
