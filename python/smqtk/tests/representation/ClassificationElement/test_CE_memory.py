import unittest

from six.moves import cPickle

import smqtk.representation.classification_element.memory


class TestMemoryClassificationElement (unittest.TestCase):

    def test_serialization(self):
        e = smqtk.representation.classification_element.memory\
            .MemoryClassificationElement('test', 0)

        e2 = cPickle.loads(cPickle.dumps(e))
        self.assertEqual(e, e2)

        e.set_classification(a=0, b=1)
        e2 = cPickle.loads(cPickle.dumps(e))
        self.assertEqual(e, e2)
