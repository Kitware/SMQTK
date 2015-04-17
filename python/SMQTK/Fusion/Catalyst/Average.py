# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from SMQTK.Fusion.Catalyst import Catalyst


class Average (Catalyst):
    """
    Fuze by averaging probabilities for a given UID
    """

    def fuse(self, *rank_maps):
        """
        Given one or more dictionaries mapping UID to probability value (value
        between [0,1]), fuse by some manner unto a single comprehensive mapping
        of UID to probability value.

        >>> c = Average('unused_data', 'unused_work')
        >>> rm_1 = {0: 0., \
                    1: 0., \
                    2: 0.}
        >>> rm_2 = {0: 1., \
                    1: 1., \
                    2: 1.}
        >>> rm_final = c.fuse(rm_1, rm_2)
        >>> assert rm_final[0] == 0.5
        >>> assert rm_final[1] == 0.5
        >>> assert rm_final[2] == 0.5

        :raises ValueError: No ranking maps given.

        :param rank_maps: One or more rank dictionaries
        :type rank_maps: list of (dict of (int, float))

        :return: Fused ranking dictionary
        :rtype: dict of (int, float)

        """
        super(Average, self).fuse(*rank_maps)

        all_uids = set()
        for rm in rank_maps:
            all_uids.update(rm.keys())

        final = {}
        for uid in all_uids:
            i = 0
            v = 0
            for rm in rank_maps:
                if uid in rm:
                    v += rm[uid]
                    i += 1
            final[uid] = v / float(i)

        return final

    def reset(self):
        """
        Reset this catalyst instance to its original state.

        This implementation has no model to reset, so this method is a no-op.

        """
        return


CATALYST_CLASS = Average
