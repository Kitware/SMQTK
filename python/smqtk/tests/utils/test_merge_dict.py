import unittest

from smqtk.utils.dict import merge_dict


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
        self.assertEqual(self.a, self.expected)

        # set values that are mutable structures should be the same instances as
        # what's in ``b``.
        self.assertEqual(self.a['nested']['l'],
                         self.b['nested']['l'])
        self.assertIs(self.a['nested']['l'],
                      self.b['nested']['l'])

        self.assertEqual(self.a['nested']['even_deeper']['j'],
                         self.b['nested']['even_deeper']['j'])
        self.assertIs(self.a['nested']['even_deeper']['j'],
                      self.b['nested']['even_deeper']['j'])

    def test_merge_dict_deepcopy(self):
        # dict merger with deepcopy
        merge_dict(self.a, self.b, deep_copy=True)
        self.assertEqual(self.a, self.expected)

        # set values that are mutable structures should be the same instances as
        # what's in ``b``.
        self.assertEqual(self.a['nested']['l'],
                         self.b['nested']['l'])
        self.assertIsNot(self.a['nested']['l'],
                         self.b['nested']['l'])

        self.assertEqual(self.a['nested']['even_deeper']['j'],
                         self.b['nested']['even_deeper']['j'])
        self.assertIsNot(self.a['nested']['even_deeper']['j'],
                         self.b['nested']['even_deeper']['j'])

    def test_merge_dict_return_a(self):
        # Return value should be the ``a`` parameter input value
        r = merge_dict(self.a, self.b)
        self.assertIs(r, self.a)

    def test_disjoint_update(self):
        """ Test that ``'c': 3`` gets added to `a`. """
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
        self.assertEqual(a, expected)

    def test_subset_merge(self):
        """ Test that `b` updates key `a` in dict `a`. """
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
        self.assertEqual(a, expected)

    def test_partial_update(self):
        """
        Test that dict `b` updates and adds keys appropriately to dict `a` (adds
        key `c`, updates key `a`).
        """
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
        self.assertEqual(a, expected)

    def test_overrides(self):
        """ Test that dict `b` overrides key `b` with a nested dict. """
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
        self.assertEqual(a, expected)

    def test_nested(self):
        """
        Test that nested dictionary updates occur correctly without losing
        values in sibling or nephew dictionaries.
        """
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
        self.assertEqual(a, expected)
