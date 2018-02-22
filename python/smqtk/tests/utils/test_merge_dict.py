from __future__ import division, print_function
import unittest

import nose.tools

from smqtk.utils import merge_dict


class TestMergeDict (unittest.TestCase):

    def setUp(self):
        # Reset values
        self.a = {
            1: 2,
            3: 'value',
            'nested': {
                'v': 'sub-value',
                'l': [0, 1, 2],
                'i': 4,
                'even_deeper': {
                    'j': 'x',
                    'k': 'l',
                },
            },
            'nested2': {
                'a': 'will be overwritten'
            },
            's': {'some', 'set', 'here'}
        }

        self.b = {
            3: 'different_value',
            4: 'new value',
            'nested': {
                'l': [3, 4, 5],
                'i': 5,
                'new': 'pair',
                'even_deeper': {
                    'j': {
                        'new': 'sub-dict'
                    },
                },
            },
            'nested2': 'overwritten value',
        }

        self.expected = {
            1: 2,
            3: 'different_value',
            4: 'new value',
            'nested': {
                'v': 'sub-value',
                'l': [3, 4, 5],
                'i': 5,
                'new': 'pair',
                'even_deeper': {
                    'j': {
                        'new': 'sub-dict'
                    },
                    'k': 'l',
                },
            },
            'nested2': 'overwritten value',
            's': {'some', 'set', 'here'}
        }

    def test_merge_dict_shallow(self):
        # basic dict merger
        merge_dict(self.a, self.b)
        nose.tools.assert_equal(self.a, self.expected)

        # set values that are mutable structures should be the same instances as
        # what's in ``b``.
        nose.tools.assert_equal(self.a['nested']['l'],
                                self.b['nested']['l'])
        nose.tools.assert_is(self.a['nested']['l'],
                             self.b['nested']['l'])

        nose.tools.assert_equal(self.a['nested']['even_deeper']['j'],
                                self.b['nested']['even_deeper']['j'])
        nose.tools.assert_is(self.a['nested']['even_deeper']['j'],
                             self.b['nested']['even_deeper']['j'])

    def test_merge_dict_deepcopy(self):
        # dict merger with deepcopy
        merge_dict(self.a, self.b, deep_copy=True)
        nose.tools.assert_equal(self.a, self.expected)

        # set values that are mutable structures should be the same instances as
        # what's in ``b``.
        nose.tools.assert_equal(self.a['nested']['l'],
                                self.b['nested']['l'])
        nose.tools.assert_is_not(self.a['nested']['l'],
                                 self.b['nested']['l'])

        nose.tools.assert_equal(self.a['nested']['even_deeper']['j'],
                                self.b['nested']['even_deeper']['j'])
        nose.tools.assert_is_not(self.a['nested']['even_deeper']['j'],
                                 self.b['nested']['even_deeper']['j'])

    def test_merge_dict_return_a(self):
        # Return value should be the ``a`` parameter input value
        r = merge_dict(self.a, self.b)
        nose.tools.assert_is(r, self.a)
