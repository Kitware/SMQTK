import cPickle
import unittest

import nose.tools

import smqtk.representation.classification_element.memory


__author__ = "paul.tunison@kitware.com"


class TestMemoryClassificationElement (unittest.TestCase):

    def test_serialization(self):
        e = smqtk.representation.classification_element.memory\
            .MemoryClassificationElement('test', 0)

        e2 = cPickle.loads(cPickle.dumps(e))
        nose.tools.assert_equal(e, e2)

        e.set_classification(a=0, b=1)
        e2 = cPickle.loads(cPickle.dumps(e))
        nose.tools.assert_equal(e, e2)
