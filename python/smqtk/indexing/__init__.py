"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import os


class Indexer (object):
    """
    Base class for indexer implementations.

    Indexers are responsible for:
        - Generating a data model given an ingest.
        - Add new data to an existing data model.
        - Rank the the content of the indexer's model given positive and
            negative exemplars.

    """
    __metaclass__ = abc.ABCMeta

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @abc.abstractmethod
    def generate_model(self, descriptor_map, parallel=None, **kwargs):
        """
        Generate this indexers data-model using the given features,
        saving it to files in the configured data directory.

        :raises ValueError: The given feature map had no content.

        :param descriptor_map: Mapping of hashable IDs to descriptor data. All
            descriptor vector data must be of the same size!
        :type descriptor_map: dict[collections.Hashable,
                                   smqtk.data_rep.DescriptorElement]

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        return

    @abc.abstractmethod
    def extend_model(self, uid_feature_map, parallel=None):
        """
        Extend, in memory, the current model with the given feature elements.
        Online extensions are not saved to data files.

        NOTE: For now, if there is currently no data model created for this
        indexer / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: Unable to expand model.

        :raises ValueError: See implementation.

        :param uid_feature_map: Mapping of integer IDs to features to extend this
            indexer's model with.
        :type uid_feature_map: dict of (collections.Hashable, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores. Not all implementation support parallel model extension.
        :type parallel: int

        """
        return

    @abc.abstractmethod
    def rank(self, pos_ids, neg_ids=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1], where 1.0 is the highest rank and 0.0 is the lowest rank.

        :raises RuntimeError: Unable to produce a rank mapping.

        :param pos_ids: List of positive data IDs
        :type pos_ids: collections.Iterable of int

        :param neg_ids: List of negative data IDs
        :type neg_ids: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        return

    @abc.abstractmethod
    def reset(self):
        """
        Reset this indexer to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset.

        """
        return


def get_indexers(reload_modules=False):
    """
    Discover and return Indexer classes found in the given plugin search
    directory. Keys in the returned map are the names of the discovered classes,
    and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``INDEXER_CLASS``, which can either be a single class object or an iterable
    of class objects, to be exported. If the variable is set to None, we skip
    that module and do not import anything. If the variable is not present, we
    look for a class by the same name and casing as the module. If neither are
    found, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``Indexer`` whose keys are
        the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "INDEXER_CLASS"
    return get_plugins(__name__, this_dir, helper_var, Indexer, None,
                       reload_modules)
