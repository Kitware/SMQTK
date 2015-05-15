# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging


class Atom (object):
    """
    A single Descriptor/Indexer unit pairing.
    """

    def __init__(self, descriptor, indexers):
        """
        :param descriptor: ContentDescriptor instance to contain
        :type descriptor: smqtk.content_description.ContentDescriptor

        :param indexers: One or more indexer instances to associate with the
            provided descriptor (i.e. have models generated with the given
            descriptor's features).
        :type indexers: collections.Iterable of smqtk.Indexers.Indexer

        """
        self._descriptor = descriptor
        self._indexer_list = list(indexers)

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
        Extend this atom's indexer mode for the given data element. The
        contained descriptor generated the necessary feature.

        :param data: Data element(s) to extend the index with
        :type data: list of SMQTK.utils.DataFile.DataFile

        """
        self.log.debug("Computing features for data")
        feature_map = {}
        for d in data:
            feature_map[d.uid] = self._descriptor.compute_feature(d)
        self.log.debug("Extending indexer models")
        for i in self._indexer_list:
            i.extend_model(feature_map)

    def rank(self, pos, neg=()):
        """
        Perform ranking for all indexers present in this atom. Returns a list of
        ranking dictionaries (see documentation for
        smqtk.indexing.Indexer.rank_model) in the same number and order as
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
