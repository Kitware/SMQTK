import unittest

import mock

from smqtk.algorithms.classifier import ClassifierCollection
from smqtk.exceptions import MissingLabelError
from smqtk.representation.classification_element.memory \
    import MemoryClassificationElement
from smqtk.representation.descriptor_element.local_elements \
    import DescriptorMemoryElement
from smqtk.tests.algorithms.classifier.test_ClassifierAbstract \
    import DummyClassifier


class TestClassifierCollection (unittest.TestCase):

    ##########################################################################
    # Constructor Tests

    def test_new_empty(self):
        ccol = ClassifierCollection()
        self.assertEqual(ccol._label_to_classifier, {})

    def test_new_not_classifier_positional(self):
        # First invalid key should be in error message.
        self.assertRaisesRegexp(
            ValueError,
            "for key 'some label'",
            ClassifierCollection,
            classifiers={'some label': 0}
        )

    def test_new_not_classifier_kwarg(self):
        # First invalid key should be in error message.
        self.assertRaisesRegexp(
            ValueError,
            "for key 'some_label'",
            ClassifierCollection,
            some_label=0
        )

    def test_new_positional(self):
        c = DummyClassifier()
        ccol = ClassifierCollection(classifiers={'a label': c})
        self.assertEqual(ccol._label_to_classifier, {'a label': c})

    def test_new_kwargs(self):
        c = DummyClassifier()
        ccol = ClassifierCollection(a_label=c)
        self.assertEqual(ccol._label_to_classifier, {'a_label': c})

    def test_new_both_pos_and_kwds(self):
        c1 = DummyClassifier()
        c2 = DummyClassifier()
        ccol = ClassifierCollection({'a': c1}, b=c2)
        self.assertEqual(ccol._label_to_classifier,
                         {'a': c1, 'b': c2})

    def test_new_duplicate_label(self):
        c1 = DummyClassifier()
        c2 = DummyClassifier()
        self.assertRaisesRegexp(
            ValueError,
            "Duplicate classifier label 'c'",
            ClassifierCollection,
            {'c': c1},
            c=c2
        )

    ##########################################################################
    # Configuration Tests

    def test_get_default_config(self):
        # Returns a non-empty dictionary with just the example key. Contains
        # a sub-dictionary that would container the implementation
        # specifications.
        c = ClassifierCollection.get_default_config()

        # Should just contain the default example
        self.assertEqual(len(c), 1)
        self.assertIn('__example_label__', c.keys())
        # Should be a plugin config after this.
        self.assertIn('type', c['__example_label__'])

    def test_get_config_empty(self):
        # The config coming out of an empty collection should be an empty
        # dictionary.
        ccol = ClassifierCollection()
        self.assertEqual(ccol.get_config(), {})

    def test_get_config_with_stuff(self):
        c1 = DummyClassifier()
        c2 = DummyClassifier()
        ccol = ClassifierCollection({'a': c1}, b=c2)
        # dummy returns {} config.
        self.assertEqual(
            ccol.get_config(),
            {
                'a': {'DummyClassifier': {}, 'type': 'DummyClassifier'},
                'b': {'DummyClassifier': {}, 'type': 'DummyClassifier'},
            }
        )

    def test_from_config_empty(self):
        ccol = ClassifierCollection.from_config({})
        self.assertEqual(ccol._label_to_classifier, {})

    def test_from_config_skip_example_key(self):
        # If the default example is left in the config, it should be skipped.
        # The string chosen for the example key should be unlikely to be used
        # in reality.
        ccol = ClassifierCollection.from_config({
            '__example_label__':
                'this should be skipped regardless of content'
        })
        self.assertEqual(ccol._label_to_classifier, {})

    @mock.patch('smqtk.algorithms.classifier._classifier_collection'
                '.get_classifier_impls')
    def test_from_config_with_content(self, m_get_impls):
        # Mocking implementation getter to only return the dummy
        # implementation.
        m_get_impls.side_effect = lambda: {
            'DummyClassifier': DummyClassifier,
        }

        ccol = ClassifierCollection.from_config({
            'a': {'DummyClassifier': {}, 'type': 'DummyClassifier'},
            'b': {'DummyClassifier': {}, 'type': 'DummyClassifier'},
        })
        self.assertEqual(
            # Using sort because return from ``keys()`` has no guarantee on
            # order.
            sorted(ccol._label_to_classifier.keys()), ['a', 'b']
        )
        self.assertIsInstance(ccol._label_to_classifier['a'], DummyClassifier)
        self.assertIsInstance(ccol._label_to_classifier['b'], DummyClassifier)

    ##########################################################################
    # Accessor Method Tests

    def test_size_len(self):
        ccol = ClassifierCollection()
        self.assertEqual(ccol.size(), 0)
        self.assertEqual(len(ccol), 0)

        ccol = ClassifierCollection(
            a=DummyClassifier(),
            b=DummyClassifier(),
        )
        self.assertEqual(ccol.size(), 2)
        self.assertEqual(len(ccol), 2)

    def test_labels_empty(self):
        ccol = ClassifierCollection()
        self.assertEqual(ccol.labels(), set())

    def test_labels(self):
        ccol = ClassifierCollection(
            classifiers={
                'b': DummyClassifier(),
            },
            a=DummyClassifier(),
            label2=DummyClassifier(),
        )
        self.assertEqual(ccol.labels(), {'a', 'b', 'label2'})

    def test_add_classifier_not_classifier(self):
        # Attempt adding a non-classifier instance
        ccol = ClassifierCollection()
        # The string 'b' is not a classifier instance.
        self.assertRaisesRegexp(
            ValueError,
            "Not given a Classifier instance",
            ccol.add_classifier,
            'a', 'b'
        )

    def test_add_classifier_duplicate_label(self):
        ccol = ClassifierCollection(a=DummyClassifier())
        self.assertRaisesRegexp(
            ValueError,
            "Duplicate label provided: 'a'",
            ccol.add_classifier,
            'a', DummyClassifier()
        )

    def test_add_classifier(self):
        ccol = ClassifierCollection()
        self.assertEqual(ccol.size(), 0)

        c = DummyClassifier()
        ccol.add_classifier('label', c)
        self.assertEqual(ccol.size(), 1)
        self.assertEqual(ccol._label_to_classifier['label'], c)

    def test_get_classifier_bad_label(self):
        c = DummyClassifier()
        ccol = ClassifierCollection(a=c)
        self.assertRaises(
            KeyError,
            ccol.get_classifier,
            'b'
        )

    def test_get_classifier(self):
        c = DummyClassifier()
        ccol = ClassifierCollection(a=c)
        self.assertEqual(ccol.get_classifier('a'), c)

    def test_remove_classifier_bad_label(self):
        c = DummyClassifier()
        ccol = ClassifierCollection(a=c)
        self.assertRaises(
            KeyError,
            ccol.remove_classifier, 'b'
        )

    def test_remove_classifier(self):
        c = DummyClassifier()
        ccol = ClassifierCollection(a=c)
        ccol.remove_classifier('a')
        self.assertEqual(ccol._label_to_classifier, {})

    ##########################################################################
    # Classification Method Tests

    def test_classify(self):
        ccol = ClassifierCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('memory', '0')
        d.set_vector(d_v)
        result = ccol.classify(d)

        # Should contain one entry for each configured classifier.
        self.assertEqual(len(result), 2)
        self.assertIn('subjectA', result)
        self.assertIn('subjectB', result)
        # Each key should map to a classification element (memory in this case
        # because we're using the default factory)
        self.assertIsInstance(result['subjectA'], MemoryClassificationElement)
        self.assertIsInstance(result['subjectB'], MemoryClassificationElement)
        # We know the dummy classifier outputs "classifications" in a
        # deterministic way: class label is descriptor UUID and classification
        # value is its vector as a list.
        self.assertDictEqual(result['subjectA'].get_classification(),
                             {'0': d_v})
        self.assertDictEqual(result['subjectB'].get_classification(),
                             {'0': d_v})

    def test_classify_subset(self):
        ccol = ClassifierCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        classifierB = ccol._label_to_classifier['subjectB']
        classifierB.classify = mock.Mock()

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('memory', '0')
        d.set_vector(d_v)
        result = ccol.classify(d, labels=['subjectA'])

        # Should contain one entry for each requested classifier.
        self.assertEqual(len(result), 1)
        self.assertIn('subjectA', result)
        self.assertNotIn('subjectB', result)
        self.assertFalse(classifierB.classify.called)
        # Each key should map to a classification element (memory in this case
        # because we're using the default factory)
        self.assertIsInstance(result['subjectA'], MemoryClassificationElement)
        # We know the dummy classifier outputs "classifications" in a
        # deterministic way: class label is descriptor UUID and classification
        # value is its vector as a list.
        self.assertDictEqual(result['subjectA'].get_classification(),
                             {'0': d_v})

    def test_classify_empty_subset(self):
        ccol = ClassifierCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        classifierA = ccol._label_to_classifier['subjectA']
        classifierA.classify = mock.Mock()
        classifierB = ccol._label_to_classifier['subjectB']
        classifierB.classify = mock.Mock()

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('memory', '0')
        d.set_vector(d_v)
        result = ccol.classify(d, labels=[])

        # Should contain no entries.
        self.assertEqual(len(result), 0)
        self.assertNotIn('subjectA', result)
        self.assertFalse(classifierA.classify.called)
        self.assertNotIn('subjectB', result)
        self.assertFalse(classifierB.classify.called)

    def test_classify_missing_label(self):
        ccol = ClassifierCollection({
            'subjectA': DummyClassifier(),
            'subjectB': DummyClassifier(),
        })

        d_v = [0, 1, 2, 3, 4]
        d = DescriptorMemoryElement('memory', '0')
        d.set_vector(d_v)

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectC'])
        self.assertSetEqual(cm.exception.labels, {'subjectC'})

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectA', 'subjectC'])
        self.assertSetEqual(cm.exception.labels, {'subjectC'})

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectC', 'subjectD'])
        self.assertSetEqual(cm.exception.labels, {'subjectC', 'subjectD'})

        # Should throw a MissingLabelError
        with self.assertRaises(MissingLabelError) as cm:
            ccol.classify(d, labels=['subjectA', 'subjectC', 'subjectD'])
        self.assertSetEqual(cm.exception.labels, {'subjectC', 'subjectD'})
