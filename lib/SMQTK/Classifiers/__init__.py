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
        - Rank the the content of the ingest given positive and negative
            exemplars within the ingest.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, base_data_dir, base_work_dir, descriptor, ingest):
        """
        Initialize classifier over a given ingest.

        :param base_data_dir: Base data directory for this classifier to
            initialize to.
        :type base_data_dir: str

        :param base_work_dir: Base work directory for this classifier to
            initialize to.
        :type base_work_dir: str

        :param descriptor: A FeatureDescriptor instance for this classifier to
            use.
        :type descriptor: SMQTK.FeatureDescriptors.FeatureDescriptor

        :param ingest: Data ingest for this classifier to work over.
        :type ingest: SMQTK.utils.DataIngest.DataIngest

        """
        self._data_dir = os.path.join(base_data_dir,
                                      "Classifiers",
                                      self.__class__.__name__,
                                      descriptor.__class__.__name__,
                                      ingest.name)
        self._work_dir = os.path.join(base_work_dir,
                                      "Classifiers",
                                      self.__class__.__name__,
                                      descriptor.__class__.__name__,
                                      ingest.name)
        self._descriptor = descriptor

        self._ingest = ingest

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
    def generate_model(self):
        """
        Generate this classifiers data-model using the given feature descriptor
        over the configured ingest, saving it to a known location in the
        configured data directory.
        """
        pass

    @abc.abstractmethod
    def extend_model(self, data):
        """
        Extend the current ingest with new feature data based on the given data
        and the configured feature descriptor, extending the current data model
        for this classifier.

        For not, if there is currently no data model created for this classifier
        / descriptor combination, we will error. In the future, I would imagine
        a new model would be created.

        :param data: Some kind of input data for the feature descriptor. This is
            descriptor dependent.

        """

    @abc.abstractmethod
    def rank_ingest(self, pos_ids, neg_ids):
        """
        Rank the current ingest, returning a mapping of ingest element ID to a
        ranking valuation. This valuation should be a probability in the range
        of [0, 1]. Where

        :return: Mapping of ingest ID to a rank.
        :rtype: dict of (int, float)

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
    :rtype: dict of (str, SMQTKClassifier)

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
            fd_classes = None
            if hasattr(module, standard_var):
                fd_classes = getattr(module, standard_var, None)
                if isinstance(fd_classes, (tuple, list)):
                    log.debug('[%s] Loaded list of classes via variable: '
                              '%s',
                              module_name, fd_classes)
                elif issubclass(fd_classes, SMQTKClassifier):
                    log.debug("[%s] Loaded class via variable: %s",
                              module_name, fd_classes)
                    fd_classes = [fd_classes]
                else:
                    raise RuntimeError("[%s] %s variable not set to a "
                                       "valid value.",
                                       module_name)

            # Try finding a class with the same name as the module
            elif hasattr(module, module.__name__):
                fd_classes = getattr(module, module.__name__, None)
                if issubclass(fd_classes, SMQTKClassifier):
                    log.debug("[%s] Loaded class by module name: %s",
                              module_name, fd_classes)
                    fd_classes = [fd_classes]
                else:
                    raise RuntimeError("[%s] Failed to find valid class by "
                                       "module name",
                                       module_name)

            for cls in fd_classes:
                class_map[cls.__name__] = cls

    return class_map
