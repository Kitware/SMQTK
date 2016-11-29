import os
import unittest

import nose.tools as ntools

from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR
from smqtk.utils.image_utils import is_loadable_image, is_valid_element


class TestIsLoadableImage(unittest.TestCase):

    def setUp(self):
        self.good_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                       'Lenna.png'))
        self.non_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                      'test_file.dat'))

    @ntools.raises(AttributeError)
    def test_non_data_element_raises_exception(self):
        # should throw:
        # AttributeError: 'bool' object has no attribute 'get_bytes'
        is_loadable_image(False)


    def test_unloadable_image_returns_false(self):
        assert is_loadable_image(self.non_image) == False


    def test_unloadable_image_logs_warning(self):
        pass


    def test_loadable_image_returns_true(self):
        assert is_loadable_image(self.good_image) == True


class TestIsValidElement(unittest.TestCase):

    def setUp(self):
        self.good_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                       'Lenna.png'))
        self.non_image = DataFileElement(os.path.join(TEST_DATA_DIR,
                                                      'test_file.dat'))


    def test_non_data_element(self):
        assert is_valid_element(False) == False


    def test_invalid_content_type(self):
        # test it logs to debug
        assert is_valid_element(self.good_image, valid_content_types=[]) == False

    def test_valid_content_type(self):
        assert is_valid_element(self.good_image,
                                valid_content_types=['image/png']) == True


    def test_invalid_image_returns_false(self):
        assert is_valid_element(self.non_image, check_image=True) == False
