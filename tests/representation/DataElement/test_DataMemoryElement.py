import random
import unittest
import six

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation.data_element.memory_element import DataMemoryElement


def random_string(length):
    # 32-127 is legible characters
    return ''.join(chr(random.randint(32, 127)) for _ in range(length))


class TestDataMemoryElement (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EXPECTED_BYTES = six.b('hello world')
        cls.EXPECTED_CT = 'text/plain'

        cls.VALID_BASE64 = 'aGVsbG8gd29ybGQ='
        cls.INVALID_BASE64 = '$%&^c85swd8a5sw568vs!'

        cls.VALID_B64_URI = 'base64://' + cls.VALID_BASE64
        cls.VALID_DATA_URI = 'data:' + cls.EXPECTED_CT + ';base64,' + \
                             cls.VALID_BASE64

    def test_configuration(self):
        default_config = DataMemoryElement.get_default_config()
        self.assertEqual(default_config,
                         {'bytes': None, 'content_type': None,
                          'readonly': False})

        default_config['bytes'] = 'Hello World.'
        default_config['content_type'] = 'text/plain'
        #: :type: DataMemoryElement
        inst1 = DataMemoryElement.from_config(default_config)
        self.assertEqual(default_config, inst1.get_config())
        self.assertEqual(inst1._bytes, b'Hello World.')
        self.assertEqual(inst1._content_type, 'text/plain')

        inst2 = DataMemoryElement.from_config(inst1.get_config())
        self.assertEqual(inst1, inst2)

    #
    # from_base64 tests
    #

    def test_from_base64_no_ct(self):
        e = DataMemoryElement.from_base64(self.VALID_BASE64)
        self.assertIsInstance(e, DataMemoryElement)
        self.assertEqual(e.get_bytes(), self.EXPECTED_BYTES)

    def test_from_base64_with_ct(self):
        e = DataMemoryElement.from_base64(self.VALID_BASE64, self.EXPECTED_CT)
        self.assertIsInstance(e, DataMemoryElement)
        self.assertEqual(e.get_bytes(), self.EXPECTED_BYTES)
        self.assertEqual(e.content_type(), self.EXPECTED_CT)

    def test_from_base64_null_bytes(self):
        self.assertRaises(
            ValueError,
            DataMemoryElement.from_base64,
            None, None
        )

    def test_from_base64_empty_string(self):
        # Should translate to empty byte string
        e = DataMemoryElement.from_base64('', None)
        self.assertIsInstance(e, DataMemoryElement)
        self.assertEqual(e.get_bytes(), six.b(''))

    #
    # From URI tests
    #

    def test_from_uri_null_uri(self):
        self.assertRaises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            None
        )

    def test_from_uri_empty_string(self):
        # Should return an element with no byte data
        e = DataMemoryElement.from_uri('')
        self.assertIsInstance(e, DataMemoryElement)
        # no base64 data, which should decode to no bytes
        self.assertEqual(e.get_bytes(), six.b(''))

    def test_from_uri_random_string(self):
        rs = random_string(32)
        self.assertRaises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            rs
        )

    def test_from_uri_base64_header_empty_data(self):
        e = DataMemoryElement.from_uri('base64://')
        self.assertIsInstance(e, DataMemoryElement)
        # no base64 data, which should decode to no bytes
        self.assertEqual(e.get_bytes(), six.b(''))

    def test_from_uri_base64_header_invalid_base64(self):
        # URI base64 data contains invalid alphabet characters
        self.assertRaises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'base64://'+self.INVALID_BASE64
        )

    def test_from_uri_base64_equals_out_of_place(self):
        # '=' characters should only show up at the end of a base64 data string
        self.assertRaises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'base64://foo=bar'
        )

    def test_from_uri_base64_too_many_equals(self):
        # There should only be a max of 2 '=' characters at the end of the b64
        # data string
        self.assertRaises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'base64://foobar==='
        )

    def test_from_uri_base64_header(self):
        e = DataMemoryElement.from_uri(self.VALID_B64_URI)
        self.assertIsInstance(e, DataMemoryElement)
        self.assertEqual(e.get_bytes(), self.EXPECTED_BYTES)
        # No content type info in base64 format
        self.assertEqual(e.content_type(), None)

    def test_from_uri_data_format_empty_data(self):
        e = DataMemoryElement.from_uri('data:text/plain;base64,')
        self.assertIsInstance(e, DataMemoryElement)
        # no base64 data, which should decode to no bytes
        self.assertEqual(e.get_bytes(), six.b(''))
        self.assertEqual(e.content_type(), 'text/plain')

    def test_from_uri_data_format_invalid_base64(self):
        self.assertRaises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'data:text/plain;base64,' + self.INVALID_BASE64
        )

    def test_from_uri_data_format(self):
        e = DataMemoryElement.from_uri(self.VALID_DATA_URI)
        self.assertIsInstance(e, DataMemoryElement)
        self.assertEqual(e.get_bytes(), self.EXPECTED_BYTES)
        self.assertEqual(e.content_type(), self.EXPECTED_CT)

    #
    # Content tests
    #
    def test_is_empty_zero_bytes(self):
        e = DataMemoryElement(b'')
        self.assertTrue(e.is_empty())

    def test_is_empty_nonzero_bytes(self):
        e = DataMemoryElement(b'some bytes')
        self.assertFalse(e.is_empty())

    def test_get_bytes_none_bytes(self):
        e = DataMemoryElement()
        self.assertEqual(e.get_bytes(), six.b(''))

    def test_get_bytes_empty_bytes(self):
        e = DataMemoryElement(b'')
        self.assertEqual(e.get_bytes(), b'')

    def test_get_bytes_some_bytes(self):
        expected_bytes = b'some bytes'
        e = DataMemoryElement(expected_bytes)
        self.assertEqual(e.get_bytes(), expected_bytes)

    def test_writable_default(self):
        v = b'foo'
        e = DataMemoryElement(v)
        self.assertTrue(e.writable())

    def test_writable_when_readonly(self):
        e = DataMemoryElement(b'', readonly=True)
        self.assertFalse(e.writable())

    def test_writable_when_not_readonly(self):
        e = DataMemoryElement(b'', readonly=False)
        self.assertTrue(e.writable())

    def test_set_bytes(self):
        bytes_a = six.b('test bytes first set')
        bytes_b = six.b('the second set of bytes')
        e = DataMemoryElement(bytes_a)
        self.assertEqual(e.get_bytes(), bytes_a)
        e.set_bytes(bytes_b)
        self.assertEqual(e.get_bytes(), bytes_b)

    def test_set_bytes_when_readonly(self):
        bytes_a = six.b('test bytes first set')
        bytes_b = six.b('the second set of bytes')
        e = DataMemoryElement(bytes_a, readonly=True)
        self.assertEqual(e.get_bytes(), bytes_a)
        self.assertRaises(
            ReadOnlyError,
            e.set_bytes,
            bytes_b
        )
        self.assertEqual(e.get_bytes(), bytes_a)
