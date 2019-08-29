import threading
import unittest

import pytest
from six.moves import cPickle, mock

from smqtk.exceptions import NoClassificationError
from smqtk.representation.classification_element.memory \
    import MemoryClassificationElement


class TestMemoryClassificationElement (unittest.TestCase):

    def test_is_usable(self):
        """
        Test that this implementation is usable (should always be)
        """
        assert MemoryClassificationElement.is_usable()

    def test_init(self):
        """
        Test that construction sets the appropriate attributes.
        """
        expected_typename = 'ex typename'
        expected_uuid = 'ex uuid'

        #: :type: mock.MagicMock | MemoryClassificationElement
        m = mock.MagicMock(spec_set=MemoryClassificationElement)
        MemoryClassificationElement.__init__(m, expected_typename,
                                             expected_uuid)
        assert hasattr(m, '_c')
        assert m._c is None
        assert hasattr(m, '_c_lock')
        # in python 2, threading.RLock() is threading._RLock, but in 3 its _thread.RLock
        assert isinstance(m._c_lock, type(threading.RLock()))

    def test_serialization_empty(self):
        e = MemoryClassificationElement('test', 0)
        # Keep it empty.

        expected_map = None
        assert e._c == expected_map
        e2 = cPickle.loads(cPickle.dumps(e))
        assert e2._c == expected_map

    def test_serialization_nonempty(self):
        e = MemoryClassificationElement('test', 0)
        e.set_classification(a=0, b=1)

        expected_map = {'a': 0, 'b': 1}
        assert e._c == expected_map
        e2 = cPickle.loads(cPickle.dumps(e))
        assert e2._c == expected_map

    def test_get_config_empty(self):
        """
        Test that configuration returned is empty.
        """
        e = MemoryClassificationElement('test', 0)
        assert e.get_config() == {}

    def test_get_config_nonempty(self):
        """
        Test that configuration returned is empty.
        """
        e = MemoryClassificationElement('test', 0)
        e._c = {'a': 1.0, 'b': 0.0}
        assert e.get_config() == {}

    def test_has_classifications_empty(self):
        """
        Test that has_classifications returns false when the internal map
        has either not been set or is an empty dictionary.
        """
        e = MemoryClassificationElement('test', 0)
        e._c = None
        assert e.has_classifications() is False
        e._c = {}
        assert e.has_classifications() is False

    def test_has_classification_nonempty(self):
        """
        Test that has_classifications returns true when there is a valid
        internal map.
        """
        e = MemoryClassificationElement('test', 0)
        e._c = {'a': 1, 'b': 0}
        assert e.has_classifications() is True

    def test_get_classification_empty(self):
        """
        Test that NoClassificationError is raised when there is no or an empty
        classification map set.
        """
        e = MemoryClassificationElement('test', 0)
        e._c = None
        with pytest.raises(NoClassificationError,
                           match="No classification labels/values"):
            e.get_classification()

    def test_get_classification(self):
        """
        Test that a valid classification map is returned from
        """
        expected_map = {'a': 1, 'b': 0}
        e = MemoryClassificationElement('test', 0)
        e._c = expected_map
        assert e.get_classification() == expected_map

    def test_set_classification(self):
        """
        Test setting valid classification map.
        """
        e = MemoryClassificationElement('test', 0)

        expected_map = {'a': 1, 'b': 0}
        e.set_classification(expected_map)
        assert e._c == expected_map
