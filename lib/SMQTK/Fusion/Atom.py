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

    def __init__(self, descriptor, indexers):
        """
        :param descriptor: FeatureDescriptor instance to contain
        :type descriptor: SMQTK.FeatureDescriptors.FeatureDescriptor

        :param indexers: One or more indexer instances to associate with the
            provided descriptor (i.e. have models generated with the given
            descriptor's features).
        :type indexers: collections.Iterable of SMQTK.Indexers.Indexer

        """
        self._descriptor = descriptor
        self._indexer_list = list(indexers)

    def extend(self, data):
        """
        Extend this atom's indexer mode for the given data element. The
        contained descriptor generated the necessary feature.

        :param data: Data element to extend the index with
        :type data: SMQTK.utils.DataFile.DataFile

        """
        feature = self._descriptor.compute_feature(data)
        for i in self._indexer_list:
            i.extend_model({data.uid: feature})

    def rank(self, pos, neg=()):
        """
        Perform ranking for all indexers present in this atom. Returns a list of
        ranking dictionaries (see documentation for
        SMQTK.Indexers.Indexer.rank_model) in the same number and order as
        indexers were provided to this atom during instantiation.

        :raises RuntimeError: No current model in one or more indexers.

        :param pos: List of positive data IDs
        :type pos: collections.Iterable of int

        :param neg: List of negative data IDs
        :type neg: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: list of (dict of (int, float))

        """
        rank_results = []
        for i in self._indexer_list:
            rank_results.append(i.rank_model(pos, neg))
        return rank_results

    def reset(self):
        """
        Reset this atom to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        for i in self._indexer_list:
            i.reset()
