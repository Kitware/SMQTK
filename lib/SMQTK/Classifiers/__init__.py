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
import re


class SMQTKClassifier (object):
    """
    Base class for classifier implementations.

    Classifiers are responsible for:
        - Generating a data model given an ingest.
        - Add new data to an existing data model.
        - Rank the the content of the classifier's model given positive and
            negative exemplar IDs.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, work_dir):
        """
        Initialize classifier with a given descriptor instance.

        Construction of multiple classifier instances is expected to involve
        providing a similar data directory but different work directories. The
        data directory would only be read from except for when generating a
        model which would error if there was already something there (read-only
        enforcement).

        :param data_dir: Classifier data directory
        :type data_dir: str

        :param work_dir: Work directory for this classifier to use.
        :type work_dir: str

        """
        self._data_dir = data_dir
        self._work_dir = work_dir

    @property
    def name(self):
        """
        :return: Classifier type name
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
        :return: This classifier type's base data directory
        :rtype: str
        """
        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        return self._data_dir

    @property
    def work_dir(self):
        """
        :return: This classifier type's base work directory
        :rtype: str
        """
        if not os.path.isdir(self._work_dir):
            os.makedirs(self._work_dir)
        return self._work_dir

    @abc.abstractmethod
    def generate_model(self, feature_map, parallel=None):
        """
        Generate this classifiers data-model using the given feature descriptor
        over the configured ingest, saving it to a known location in the
        configured data directory.

        :raises RuntimeError: See implementation.

        :param feature_map: Mapping of integer IDs to feature data. All feature
            data must be of the same size!
        :type feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        pass

    @abc.abstractmethod
    def extend_model(self, id_feature_map, parallel=None):
        """
        Extend, in memory, the current model with the given data elements using
        the configured feature descriptor.

        NOTE: For now, if there is currently no data model created for this
        classifier / descriptor combination, we will error. In the future, I
        would imagine a new model would be created.

        :raises RuntimeError: See implementation.
        :raises ValueError: See implementation.

        :param id_feature_map: Mapping of integer IDs to features to extend this
            classifier's model with.
        :type id_feature_map: dict of (int, numpy.core.multiarray.ndarray)

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores. Not all implementation support parallel model extension.
        :type parallel: int

        """
        pass

    @abc.abstractmethod
    def rank_model(self, pos_ids, neg_ids=()):
        """
        Rank the current model, returning a mapping of element IDs to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1]. Where

        :param pos_ids: List of positive data IDs
        :type pos_ids: collections.Iterable of int

        :param neg_ids: List of negative data IDs
        :type neg_ids: collections.Iterable of int

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset this classifier to its original state, i.e. removing any model
        extension that may have occurred.

        :raises RuntimeError: See implementation.

        """
        pass


def get_classifiers():
    """
    Discover and return SMQTKClassifier classes found in the fixed plugin
    directory. Keys will be the name of the discovered SMQTKClassifier class
    with the paired value being the associated class object.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files are "hidden").

    Within the module we look first for a variable named
    "CLASSIFIER_CLASS", which can either be a class object or a list of
    class objects, to be exported. If the above variable is not found, we look
    for a class by the same name of the module. If neither are found, we raise
    a RuntimeError.

    :return: Map of discovered SMQTKClassifier types whose keys are the string
        name of the class.
    :rtype: dict of (str, type)

    """
    log = logging.getLogger("get_classifers")
    class_map = {}

    this_dir = os.path.abspath(os.path.dirname(__file__))
    log.debug("Searching in directory: %s", this_dir)

    file_re = re.compile("^[a-zA-Z].*(?:\.py)?$")
    standard_var = "CLASSIFIER_CLASS"

    for module_file in os.listdir(this_dir):
        if file_re.match(module_file):
            log.debug("Examining file: %s", module_file)

            module_name = os.path.splitext(module_file)[0]

            module_path = '.'.join([__name__, module_name])
            log.debug("Attempting import of: %s", module_path)
            module = __import__(module_path, fromlist=__name__)

            # Look for standard variable
            cl_classes = []
            if hasattr(module, standard_var):
                cl_classes = getattr(module, standard_var, None)
                if isinstance(cl_classes, (tuple, list)):
                    log.debug('[%s] Loaded list of classes via variable: '
                              '%s',
                              module_name, cl_classes)
                elif issubclass(cl_classes, SMQTKClassifier):
                    log.debug("[%s] Loaded class via variable: %s",
                              module_name, cl_classes)
                    cl_classes = [cl_classes]
                else:
                    raise RuntimeError("[%s] %s variable not set to a "
                                       "valid value.",
                                       module_name)

            # Try finding a class with the same name as the module
            elif hasattr(module, module.__name__):
                cl_classes = getattr(module, module.__name__, None)
                if issubclass(cl_classes, SMQTKClassifier):
                    log.debug("[%s] Loaded class by module name: %s",
                              module_name, cl_classes)
                    cl_classes = [cl_classes]
                else:
                    raise RuntimeError("[%s] Failed to find valid class by "
                                       "module name",
                                       module_name)

            for cls in cl_classes:
                class_map[cls.__name__] = cls

    return class_map
