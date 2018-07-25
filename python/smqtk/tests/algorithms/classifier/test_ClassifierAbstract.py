from __future__ import division, print_function
import mock
import unittest

from smqtk.algorithms.classifier import Classifier, SupervisedClassifier, \
    get_classifier_impls
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement


class DummyClassifier (Classifier):

    EXPECTED_LABELS = ['constant']

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def get_labels(self):
        return self.EXPECTED_LABELS

    def _classify(self, d):
        """
        Returns one "label" that is the UUID of the descriptor, mapping to the
        descriptor's vector as a list.
        """
        return {d.uuid(): d.vector().tolist()}


class TestClassifierAbstractClass (unittest.TestCase):

    def test_classify(self):
        d = DescriptorMemoryElement('test', 0)
        d.set_vector([1, 2, 3])

        c = DummyClassifier()
        e = c.classify(d)
        self.assertEqual(e.get_classification(), {
            0: [1, 2, 3]
        })
        self.assertEqual(e.uuid, d.uuid())

    def test_classify_empty_descriptor(self):
        d = DescriptorMemoryElement('test', 0)
        c = DummyClassifier()
        self.assertRaisesRegexp(
            ValueError,
            "does not contain a vector to classify",
            c.classify, d
        )

    @mock.patch('smqtk.algorithms.classifier._defaults'
                '.MemoryClassificationElement.has_classifications')
    def test_classify_no_overwrite(self, m_ce_hc):
        # Testing logic when classifier element for descriptor already has
        # stored results and we are NOT overwriting. Mocking classification
        # element used in association with default element factory.
        m_ce_hc.return_value = True

        c = DummyClassifier()
        # Record if underlying classify call invoked
        c._classify = mock.MagicMock(side_effect=c._classify)

        d = DescriptorMemoryElement('test', 0)
        d.set_vector([1, 2, 3])

        c.classify(d, overwrite=False)
        c._classify.assert_not_called()

    @mock.patch('smqtk.algorithms.classifier._defaults'
                '.MemoryClassificationElement.has_classifications')
    def test_classify_with_overwrite(self, m_ce_hc):
        # Testing logic when classification element for descriptor already has
        # stored results but we call WITH overwrite on. Mocking classification
        # element used in association with default element factory.
        m_ce_hc.return_value = True

        c = DummyClassifier()
        # Record if underlying classify call invoked
        c._classify = mock.MagicMock(side_effect=c._classify)

        d = DescriptorMemoryElement('test', 0)
        d.set_vector([1, 2, 3])

        c.classify(d, overwrite=True)
        c._classify.assert_called_once_with(d)

    def test_classify_async(self):
        # Check that async classify calls classify on all input elements
        # correctly

        # make some descriptor elements
        d_elems = []
        for i in range(20):
            d = DescriptorMemoryElement('test', i)
            d.set_vector([i])
            d_elems.append(d)

        c = DummyClassifier()
        m = c.classify_async(d_elems)

        # Due to mocking, we expect to see a map of descriptor element to the
        # result of classification, which in this case is the UUID of the
        # element
        for d in d_elems:
            self.assertIn(d, m)
            # Check for expected classification
            self.assertEqual(m[d].get_classification(),
                             {d.uuid(): d.vector().tolist()})


class TestGetClassifierImpls (unittest.TestCase):
    # IndexLabelClassifier should always be available (no dependencies)

    def test_get_classifier_impls_subclass_interface_classifier(self):
        # Simple test that primarily makes sure the function doesn't fall down.
        m = get_classifier_impls()
        self.assertIsInstance(m, dict)
        self.assertIn("IndexLabelClassifier", m)

        # Should act the same as calling with no sub-interface
        m2 = get_classifier_impls(sub_interface=Classifier)
        self.assertIsInstance(m, dict)
        self.assertIn("IndexLabelClassifier", m)
        self.assertEqual(m, m2)

    def test_get_classifier_impls_subclass_interface_supervised(self):
        # should not return when sub_interface is set to the
        # SupervisedClassifier base-class.
        m = get_classifier_impls(sub_interface=SupervisedClassifier)
        self.assertIsInstance(m, dict)
        self.assertNotIn("IndexLabelClassifier", m)
