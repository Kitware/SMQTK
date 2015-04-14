# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging


class Reactor (object):
    """
    Perform synchronous computation and fusion across multiple atoms and a
    catalyst fusor.
    """

    def __init__(self, atoms, catalyst, atom_catalysts=None):
        """
        :param atoms: Iterable of atoms to function over
        :type atoms: collections.Iterable of SMQTK.Fusion.Atom.Atom

        :param catalyst: Catalyst implementation instance
        :type catalyst: SMQTK.Fusion.Catalyst.Catalyst

        :param atom_catalysts: Optional list catalyst instances to apply
            directly to the local output of a single atom. This must be the same
            length as the list of atoms provided. None may be provided in this
            list to indicate that the atom in the associated index should not
            have a catalyst applied to its output. The results of these
            "sub-fusions", plus any atom results that didn't go through a
            sub-catalyst, are input to the final catalyst.
        :type atom_catalysts:
            collections.Iterable of (None or SMQTK.Fusion.Catalyst.Catalyst)

        """
        #: :type: tuple of SMQTK.Fusion.Atom.Atom
        self._atom_list = tuple(atoms)
        self._catalyst = catalyst

        if atom_catalysts:
            #: :type: tuple of (None or SMQTK.Fusion.Catalyst.Catalyst)
            self._atom_catalysts = tuple(atom_catalysts)
        else:
            #: :type: tuple of (None or SMQTK.Fusion.Catalyst.Catalyst)
            self._atom_catalysts = (None,) * len(self._atom_list)

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    def extend(self, *data):
        """
        Serially extend atoms with the given data element.

        :param data: Data element(s) to extend the index with
        :type data: list of SMQTK.utils.DataFile.DataFile

        """
        for a in self._atom_list:
            a.extend(*data)

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
        for a, c in zip(self._atom_list, self._atom_catalysts):
            if c:
                atom_ranks.append(c.fuze(*a.rank(pos, neg)))
            else:
                atom_ranks.extend(a.rank(pos, neg))
        return self._catalyst.fuse(*atom_ranks)

    def reset(self):
        """
        Reset atoms and catalyst to its original states.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        for a in self._atom_list:
            a.reset()
        for c in self._atom_catalysts:
            c and c.reset()
        self._catalyst.reset()
