import unittest

import nose.tools as ntools

from smqtk.utils import (
    merge_dict
)


__author__ = "paul.tunison@kitware.com"


class TestMergeConfigs (unittest.TestCase):

    def test_disjoint_update(self):
        a = {
            'a': 1,
            'b': 2,
        }
        b = {
            'c': 3
        }
        expected = {
            'a': 1,
            'b': 2,
            'c': 3,
        }
        merge_dict(a, b)
        ntools.assert_equal(a, expected)

    def test_subset_merge(self):
        a = {
            'a': 1,
            'b': 2,
        }
        b = {
            'a': 3
        }
        expected = {
            'a': 3,
            'b': 2,
        }
        merge_dict(a, b)
        ntools.assert_equal(a, expected)

    def test_partial_update(self):
        a = {
            'a': 1,
            'b': 2,
        }
        b = {
            'a': 3,
            'c': 4,
        }
        expected = {
            'a': 3,
            'b': 2,
            'c': 4,
        }
        merge_dict(a, b)
        ntools.assert_equal(a, expected)

    def test_overrides(self):
        a = {
            'a': 1,
            'b': 2,
        }
        b = {
            'b': {
                'c': 3
            },
        }
        expected = {
            'a': 1,
            'b': {
                'c': 3,
            }
        }
        merge_dict(a, b)
        ntools.assert_equal(a, expected)

    def test_nested(self):
        a = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                },
            },
            'f': {
                'g': 4,
                'h': {
                    'i': 5
                }
            },
        }
        b = {
            'b': {'c': 6},
            'f': {'h': {'i': 7}},
            'j': 8
        }
        expected = {
            'a': 1,
            'b': {
                'c': 6,
                'd': {
                    'e': 3
                },
            },
            'f': {
                'g': 4,
                'h': {
                    'i': 7
                }
            },
            'j': 8,
        }
        merge_dict(a, b)
        ntools.assert_equal(a, expected)
