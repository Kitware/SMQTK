"""
Utility functions pertaining to python dictionaries.
"""

import copy


def merge_dict(a, b, deep_copy=False):
    """
    Merge dictionary b into dictionary a.

    This is different than normal dictionary update in that we don't bash
    nested dictionaries, instead recursively updating them.

    For congruent keys, values are are overwritten, while new keys in ``b`` are
    simply added to ``a``.

    Values are assigned (not copied) by default. Setting ``deep_copy`` causes
    values from ``b`` to be deep-copied into ``a``.

    :param a: The "base" dictionary that is updated in place.
    :type a: dict

    :param b: The dictionary to merge into ``a`` recursively.
    :type b: dict

    :param deep_copy: Optionally deep-copy values from ``b`` when assigning into
        ``a``.
    :type deep_copy: bool

    :return: ``a`` dictionary after merger (not a copy).
    :rtype: dict

    """
    for k in b:
        if k in a and isinstance(a[k], dict) and isinstance(b[k], dict):
            merge_dict(a[k], b[k], deep_copy)
        elif deep_copy:
            a[k] = copy.deepcopy(b[k])
        else:
            a[k] = b[k]
    return a
