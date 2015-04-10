# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""


class Reactor (object):
    """
    Perform synchronous computation and fusion across multiple atoms and a
    catalyst fusor.
    """

    def __init__(self, atoms, catalyst):
        """
        :param atoms: Iterable of atoms to function over
        :type atoms: collections.Iterable of SMQTK.Fusion.Atom.Atom

        :param catalyst: Catalyst implementation instance
        :type catalyst: SMQTK.Fusion.Catalyst.Catalyst

        """
        self._atom_list = atoms
        self._catalyst = catalyst

    def extend(self, data):
        """
        Extend atoms with the given data element.

        :param data: Data element to extend the index with
        :type data: SMQTK.utils.DataFile.DataFile

        """
        for a in self._atom_list:
            a.extend(data)

    def rank(self, pos, neg=()):
        """
        Fuse atom rankings into a final ranking.

        :raises RuntimeError: No current model.

        :param pos: List of positive data IDs
        :type pos: collections.Iterable of int

        :param neg: List of negative data IDs
        :type neg: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        atom_ranks = []
        for a in self._atom_list:
            atom_ranks.append(a.rank(pos, neg))
        return self._catalyst.fuse(*atom_ranks)

    def reset(self):
        """
        Reset atoms and catalyst to its original states.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        for a in self._atom_list:
            a.reset()
        self._catalyst.reset()
