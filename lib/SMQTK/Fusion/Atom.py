# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""


class Atom (object):
    """
    A single Descriptor/Indexer unit pairing.
    """

    def __init__(self, descriptor, indexer):
        """
        :param descriptor: FeatureDescriptor instance to contain
        :type descriptor: SMQTK.FeatureDescriptors.FeatureDescriptor

        :param indexer: Indexer instance to contain.
        :type indexer: SMQTK.Indexers.Indexer

        """
        self._descriptor = descriptor
        self._indexer = indexer

    def extend(self, data):
        """
        Extend this atom's indexer mode for the given data element. The
        contained descriptor generated the necessary feature.

        :param data: Data element to extend the index with
        :type data: SMQTK.utils.DataFile.DataFile

        """
        feature = self._descriptor.compute_feature(data)
        self._indexer.extend_model({data.uid: feature})

    def rank(self, pos, neg=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1], where 1.0 is the highest rank and 0.0 is the lowest rank.

        :raises RuntimeError: No current model.

        :param pos: List of positive data IDs
        :type pos: collections.Iterable of int

        :param neg: List of negative data IDs
        :type neg: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        return self._indexer.rank_model(pos, neg)

    def reset(self):
        """
        Reset this atom to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        self._indexer.reset()
