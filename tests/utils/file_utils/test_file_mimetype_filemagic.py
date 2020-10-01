import os
from unittest import TestCase

from smqtk.utils.file import file_mimetype_filemagic

from tests import TEST_DATA_DIR

try:
    import magic  # type: ignore
    # We know there are multiple modules named magic. Make sure the function we
    # expect is there.
    # noinspection PyStatementEffect
    magic.detect_from_filename
except (ImportError, AttributeError):
    magic = None


if magic is not None:

    class TestFile_mimetype_filemagic(TestCase):

        def test_file_doesnt_exist(self):
            try:
                file_mimetype_filemagic('/this/path/probably/doesnt/exist.txt')
            except IOError as ex:
                self.assertEqual(ex.errno, 2,
                                 "Expected directory IO error #2. "
                                 "Got %d" % ex.errno)

        def test_directory_provided(self):
            try:
                file_mimetype_filemagic(TEST_DATA_DIR)
            except IOError as ex:
                self.assertEqual(ex.errno, 21,
                                 "Expected directory IO error #21. "
                                 "Got %d" % ex.errno)

        def test_get_mimetype_lenna(self):
            m = file_mimetype_filemagic(os.path.join(TEST_DATA_DIR,
                                                     'Lenna.png'))
            self.assertEqual(m, 'image/png')

        def test_get_mimetype_no_extension(self):
            m = file_mimetype_filemagic(
                os.path.join(TEST_DATA_DIR, 'text_file')
            )
            self.assertEqual(m, 'text/plain')
