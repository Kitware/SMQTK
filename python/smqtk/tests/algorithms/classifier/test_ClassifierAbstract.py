from __future__ import division, print_function
import mock
import unittest

from smqtk.algorithms.classifier import Classifier
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
