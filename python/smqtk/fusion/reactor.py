# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging
import multiprocessing.pool


class Reactor (object):
    """
    Perform synchronous computation and fusion across multiple atoms and a
    catalyst fusor.
    """

    def __init__(self, atoms, catalyst, atom_catalysts=None):
        """
        :param atoms: Iterable of atoms to function over
        :type atoms: collections.Iterable of smqtk.fusion.Atom.Atom

        :param catalyst: Catalyst implementation instance
        :type catalyst: smqtk.fusion.Catalyst.Catalyst

        :param atom_catalysts: Optional list catalyst instances to apply
            directly to the local output of a single atom. This must be the same
            length as the list of atoms provided. None may be provided in this
            list to indicate that the atom in the associated index should not
            have a catalyst applied to its output. The results of these
            "sub-fusions", plus any atom results that didn't go through a
            sub-catalyst, are input to the final catalyst.
        :type atom_catalysts:
            collections.Iterable of (None or smqtk.fusion.catalyst.catalyst)

        """
        #: :type: tuple of smqtk.fusion.Atom.Atom
        self._atom_list = tuple(atoms)
        self._catalyst = catalyst

        if atom_catalysts:
            #: :type: tuple of (None or smqtk.fusion.catalyst.catalyst)
            self._atom_catalysts = tuple(atom_catalysts)
        else:
            #: :type: tuple of (None or smqtk.fusion.catalyst.catalyst)
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
        ###
        # Async
        #
        # tp = multiprocessing.pool.ThreadPool()
        # for a in self._atom_list:
        #     tp.apply_async(a.extend, args=data)
        # tp.close()
        # tp.join()

        ###
        # Sync
        #
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
        # rank dicts to fuse using global catalyst
        ranks = []

        ###
        # Async version -- buggy with because of numpy
        #
        # # async result objects yielding rank matrices
        # ar = []
        # ar2 = []
        #
        # # pairs of catalyst and async result objects that should be fused and
        # # added to ``ranks`` list
        # cr_pairs = []
        #
        # tp = multiprocessing.pool.ThreadPool()
        #
        # for a, c in zip(self._atom_list, self._atom_catalysts):
        #     if c:
        #         cr_pairs.append([c, tp.apply_async(a.rank, (pos, neg))])
        #     else:
        #         ar.append(tp.apply_async(a.rank, (pos, neg)))
        #
        # # fuse atoms that have sub-catalysts
        # for c, r in cr_pairs:
        #     ar2.append(tp.apply_async(c.fuse, args=r.get()))
        #
        # for r in ar:
        #     ranks.extend(r.get())
        # for r in ar2:
        #     ranks.append(r.get())
        #
        # tp.close()
        # tp.join()
        # del tp

        ###
        # Synchronous version
        #
        for a, c in zip(self._atom_list, self._atom_catalysts):
            if c:
                ranks.append(c.fuse(*a.rank(pos, neg)))
            else:
                ranks.extend(a.rank(pos, neg))

        return self._catalyst.fuse(*ranks)

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
