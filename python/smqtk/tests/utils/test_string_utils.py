import unittest

import nose.tools

from smqtk.utils.string_utils import random_characters


class TestRandomCharacters (unittest.TestCase):

    def test_zero_n(self):
        nose.tools.assert_equal(random_characters(0), '')

    def test_no_char_set(self):
        nose.tools.assert_raises_regexp(
            ValueError,
            "Empty char_set given",
            random_characters, 5, ()
        )

    def test_negative_n(self):
        nose.tools.assert_raises_regexp(
            ValueError,
            "n must be a positive integer",
            random_characters, -1234
        )

    def test_floating_point_n(self):
        s = random_characters(4.7)
        nose.tools.assert_equal(len(s), 4)
