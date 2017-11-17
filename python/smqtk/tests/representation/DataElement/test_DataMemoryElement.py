import nose.tools as ntools
import random
import unittest

from smqtk.exceptions import InvalidUriError, ReadOnlyError
from smqtk.representation.data_element.memory_element import DataMemoryElement


def random_string(length):
    # 32-127 is legible characters
    return ''.join(chr(random.randint(32, 127)) for _ in range(length))


class TestDataMemoryElement (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.EXPECTED_BYTES = 'hello world'
        cls.EXPECTED_CT = 'text/plain'

        cls.VALID_BASE64 = 'aGVsbG8gd29ybGQ='
        cls.INVALID_BASE64 = '$%&^c85swd8a5sw568vs!'

        cls.VALID_B64_URI = 'base64://' + cls.VALID_BASE64
        cls.VALID_DATA_URI = 'data:' + cls.EXPECTED_CT + ';base64,' + cls.VALID_BASE64

    def test_configuration(self):
        default_config = DataMemoryElement.get_default_config()
        ntools.assert_equal(default_config,
                            {'bytes': None, 'content_type': None,
                             'readonly': False})

        default_config['bytes'] = 'Hello World.'
        default_config['content_type'] = 'text/plain'
        inst1 = DataMemoryElement.from_config(default_config)
        ntools.assert_equal(default_config, inst1.get_config())
        ntools.assert_equal(inst1._bytes, 'Hello World.')
        ntools.assert_equal(inst1._content_type, 'text/plain')

        inst2 = DataMemoryElement.from_config(inst1.get_config())
        ntools.assert_equal(inst1, inst2)

    #
    # from_base64 tests
    #

    def test_from_base64_no_ct(self):
        e = DataMemoryElement.from_base64(self.VALID_BASE64)
        ntools.assert_is_instance(e, DataMemoryElement)
        ntools.assert_equal(e.get_bytes(), self.EXPECTED_BYTES)

    def test_from_base64_with_ct(self):
        e = DataMemoryElement.from_base64(self.VALID_BASE64, self.EXPECTED_CT)
        ntools.assert_is_instance(e, DataMemoryElement)
        ntools.assert_equal(e.get_bytes(), self.EXPECTED_BYTES)
        ntools.assert_equal(e.content_type(), self.EXPECTED_CT)

    def test_from_base64_null_bytes(self):
        ntools.assert_raises(
            ValueError,
            DataMemoryElement.from_base64,
            None, None
        )

    def test_from_base64_empty_string(self):
        # Should translate to empty byte string
        e = DataMemoryElement.from_base64('', None)
        ntools.assert_is_instance(e, DataMemoryElement)
        ntools.assert_equal(e.get_bytes(), '')

    #
    # From URI tests
    #

    def test_from_uri_null_uri(self):
        ntools.assert_raises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            None
        )

    def test_from_uri_empty_string(self):
        # Should return an element with no byte data
        e = DataMemoryElement.from_uri('')
        ntools.assert_is_instance(e, DataMemoryElement)
        # no base64 data, which should decode to no bytes
        ntools.assert_equal(e.get_bytes(), '')

    def test_from_uri_random_string(self):
        rs = random_string(32)
        ntools.assert_raises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            rs
        )

    def test_from_uri_base64_header_empty_data(self):
        e = DataMemoryElement.from_uri('base64://')
        ntools.assert_is_instance(e, DataMemoryElement)
        # no base64 data, which should decode to no bytes
        ntools.assert_equal(e.get_bytes(), '')

    def test_from_uri_base64_header_invalid_base64(self):
        # URI base64 data contains invalid alphabet characters
        ntools.assert_raises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'base64://'+self.INVALID_BASE64
        )

    def test_from_uri_base64_equals_out_of_place(self):
        # '=' characters should only show up at the end of a base64 data string
        ntools.assert_raises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'base64://foo=bar'
        )

    def test_from_uri_base64_too_many_equals(self):
        # There should only be a max of 2 '=' characters at the end of the b64
        # data string
        ntools.assert_raises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'base64://foobar==='
        )

    def test_from_uri_base64_header(self):
        e = DataMemoryElement.from_uri(self.VALID_B64_URI)
        ntools.assert_is_instance(e, DataMemoryElement)
        ntools.assert_equal(e.get_bytes(), self.EXPECTED_BYTES)
        # No content type info in base64 format
        ntools.assert_equal(e.content_type(), None)

    def test_from_uri_data_format_empty_data(self):
        e = DataMemoryElement.from_uri('data:text/plain;base64,')
        ntools.assert_is_instance(e, DataMemoryElement)
        # no base64 data, which should decode to no bytes
        ntools.assert_equal(e.get_bytes(), '')
        ntools.assert_equal(e.content_type(), 'text/plain')

    def test_from_uri_data_format_invalid_base64(self):
        ntools.assert_raises(
            InvalidUriError,
            DataMemoryElement.from_uri,
            'data:text/plain;base64,' + self.INVALID_BASE64
        )

    def test_from_uri_data_format(self):
        e = DataMemoryElement.from_uri(self.VALID_DATA_URI)
        ntools.assert_is_instance(e, DataMemoryElement)
        ntools.assert_equal(e.get_bytes(), self.EXPECTED_BYTES)
        ntools.assert_equal(e.content_type(), self.EXPECTED_CT)

    #
    # Content tests
    #
    def test_is_empty_zero_bytes(self):
        e = DataMemoryElement('')
        ntools.assert_true(e.is_empty())

    def test_is_empty_nonzero_bytes(self):
        e = DataMemoryElement('some bytes')
        ntools.assert_false(e.is_empty())

    def test_get_bytes_none_bytes(self):
        e = DataMemoryElement()
        ntools.assert_equal(e.get_bytes(), '')

    def test_get_bytes_empty_bytes(self):
        e = DataMemoryElement('')
        ntools.assert_equal(e.get_bytes(), '')

    def test_get_bytes_some_bytes(self):
        expected_bytes = 'some bytes'
        e = DataMemoryElement(expected_bytes)
        ntools.assert_equal(e.get_bytes(), expected_bytes)

    def test_writable_default(self):
        v = 'foo'
        e = DataMemoryElement(v)
        ntools.assert_true(e.writable())

    def test_writable_when_readonly(self):
        e = DataMemoryElement('', readonly=True)
        ntools.assert_false(e.writable())

    def test_writable_when_not_readonly(self):
        e = DataMemoryElement('', readonly=False)
        ntools.assert_true(e.writable())

    def test_set_bytes(self):
        bytes_a = 'test bytes first set'
        bytes_b = 'the second set of bytes'
        e = DataMemoryElement(bytes_a)
        ntools.assert_equal(e.get_bytes(), bytes_a)
        e.set_bytes(bytes_b)
        ntools.assert_equal(e.get_bytes(), bytes_b)

    def test_set_bytes_when_readonly(self):
        bytes_a = 'test bytes first set'
        bytes_b = 'the second set of bytes'
        e = DataMemoryElement(bytes_a, readonly=True)
        ntools.assert_equal(e.get_bytes(), bytes_a)
        ntools.assert_raises(
            ReadOnlyError,
            e.set_bytes,
            bytes_b
        )
        ntools.assert_equal(e.get_bytes(), bytes_a)
