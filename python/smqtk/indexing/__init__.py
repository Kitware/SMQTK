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

    def __init__(self, data_dir, work_dir):
        """
        Initialize indexer with a given descriptor instance.

        Construction of multiple indexer instances is expected to involve
        providing a similar data directory but different work directories. The
        data directory would only be read from except for when generating a
        model which would error if there was already something there (read-only
        enforcement).

        :param data_dir: indexer data directory
        :type data_dir: str

        :param work_dir: Work directory for this indexer to use.
        :type work_dir: str

        """
        self._data_dir = data_dir
        self._work_dir = work_dir

    @property
    def name(self):
        """
        :return: Indexer type name
        :rtype: str
        """
        return self.__class__.__name__

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @property
    def data_dir(self):
        """
        :return: This indexer type's base data directory
        :rtype: str
        """
        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        return self._data_dir

    @property
    def work_dir(self):
        """
        :return: This indexer type's base work directory
        :rtype: str
        """
        if not os.path.isdir(self._work_dir):
            os.makedirs(self._work_dir)
        return self._work_dir

    @abc.abstractmethod
    def has_model(self):
        """
        :return: True if this indexer has a valid initialized model for
            extension and ranking (or doesn't need one to perform those tasks).
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def generate_model(self, feature_map, parallel=None, **kwargs):
        """
        Generate this indexers data-model using the given features,
        saving it to files in the configured data directory.

        :raises RuntimeError: Precaution error when there is an existing data
            model for this indexer. Manually delete or move the existing
            model before computing another one.

            Specific implementations may error on other things. See the specific
            implementations for more details.

        :raises ValueError: The given feature map had no content.

        :param feature_map: Mapping of integer IDs to feature data. All feature
            data must be of the same size!
        :type feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        if self.has_model():
            raise RuntimeError(
                "\n"
                "!!! Warning !!! Warning !!! Warning !!!\n"
                "A model already exists for this indexer! "
                "Make sure that you really want to do this by moving / "
                "deleting the existing model (file(s)). Model location: "
                "%s\n"
                "!!! Warning !!! Warning !!! Warning !!!"
                % self.data_dir
            )
        if not feature_map:
            raise ValueError("The given feature_map has no content.")

    @abc.abstractmethod
    def extend_model(self, uid_feature_map, parallel=None):
        """
        Extend, in memory, the current model with the given feature elements.
        Online extensions are not saved to data files.

        NOTE: For now, if there is currently no data model created for this
        indexer / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: No current model.

            See implementation for other possible RuntimeError causes.

        :raises ValueError: See implementation.

        :param uid_feature_map: Mapping of integer IDs to features to extend this
            indexer's model with.
        :type uid_feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores. Not all implementation support parallel model extension.
        :type parallel: int

        """
        if not self.has_model():
            raise RuntimeError("No model available for this indexer.")

    @abc.abstractmethod
    def rank_model(self, pos_ids, neg_ids=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1], where 1.0 is the highest rank and 0.0 is the lowest rank.

        :raises RuntimeError: No current model.

            See implementation for other possible RuntimeError causes.

        :param pos_ids: List of positive data IDs
        :type pos_ids: collections.Iterable of int

        :param neg_ids: List of negative data IDs
        :type neg_ids: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        if not self.has_model():
            raise RuntimeError("No model available for this indexer.")

    @abc.abstractmethod
    def reset(self):
        """
        Reset this indexer to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: Unable to reset due to lack of available model.

        """
        if not self.has_model():
            raise RuntimeError("No model available for this indexer to reset "
                               "to.")


def get_indexers():
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

    :return: Map of discovered class object of type ``Indexer`` whose keys are
        the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "INDEXER_CLASS"
    return get_plugins(__name__, this_dir, helper_var, Indexer)
