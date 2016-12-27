import os.path
from unittest import TestCase

import nose.tools

from smqtk.tests import TEST_DATA_DIR
from smqtk.utils.file_utils import file_mimetype_tika

try:
    from tika import detector as tika_detector
except ImportError:
    tika_detector = None


if tika_detector is not None:

    class TestFile_mimetype_tika(TestCase):

        def test_file_doesnt_exist(self):
            try:
                file_mimetype_tika('/this/path/probably/doesnt/exist.txt')
            except IOError as ex:
                nose.tools.assert_equal(ex.errno, 2,
                                        "Expected directory IO error #2. "
                                        "Got %d" % ex.errno)

        def test_directory_provided(self):
            try:
                file_mimetype_tika(TEST_DATA_DIR)
            except IOError as ex:
                nose.tools.assert_equal(ex.errno, 21,
                                        "Expected directory IO error #21. "
                                        "Got %d" % ex.errno)

        def test_get_mimetype_lenna(self):
            m = file_mimetype_tika(
                os.path.join(TEST_DATA_DIR, 'Lenna.png')
            )
            nose.tools.assert_equal(m, 'image/png')

        def test_get_mimetype_no_extension(self):
            m = file_mimetype_tika(
                os.path.join(TEST_DATA_DIR, 'text_file')
            )
            nose.tools.assert_equal(m, 'text/plain')
