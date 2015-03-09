"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import os
import os.path as osp
import re


class FeatureDescriptor (object):
    """
    Base abstract Feature Descriptor interface
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, base_data_directory, base_work_directory):
        # Directory that permanent data for this feature descriptor will be
        # held, if any
        self._data_dir = osp.join(base_data_directory,
                                  "FeatureDescriptors",
                                  self.name)
        # Directory that work for this feature descriptor should be put. This
        # should be considered a temporary
        self._work_dir = osp.join(base_work_directory,
                                  "FeatureDescriptorWork",
                                  self.name)

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def data_directory(self):
        """
        :return: Data directory for this feature descriptor
        :rtype: str
        """
        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        return self._data_dir

    @property
    def work_directory(self):
        """
        :return: Work directory for this feature descriptor
        :rtype: str
        """
        if not os.path.isdir(self._work_dir):
            os.makedirs(self._work_dir)
        return self._work_dir

    @abc.abstractmethod
    def compute_feature(self, data):
        """
        Given some kind of data, process and return a feature vector as a Numpy
        array.

        :raises RuntimeError: Feature extraction failure of some kind.

        :param data: Some kind of input data for the feature descriptor. This is
            descriptor dependent.

        :return: Feature vector.
        :rtype: numpy.ndarray

        """
        raise NotImplementedError()


def _get_plugins(plugin_dir):
    """
    Discover and return FeatureDescriptor classes found in the given plugin
    directory. Keys will be the name of the discovered FeatureDescriptor class
    with the paired value being the associated class object.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files are "hidden").

    Within the module we look first for a variable named
    "FEATURE_DESCRIPTOR_CLASS", which can either be a class object or a list of
    class objects, to be exported. If the above variable is not found, we look
    for a class by the same name of the module. If neither are found, we raise
    a RuntimeError.

    :return: Map of discovered FeatureDescriptor types whose keys are the string
        name of the class.
    :rtype: dict of (str, FeatureDescriptor)

    """
    log = logging.getLogger("_get_plugins['%s']" % plugin_dir)
    class_map = {}

    file_re = re.compile("^[a-zA-Z].*(?:\.py)?$")
    standard_var = "FEATURE_DESCRIPTOR_CLASS"

    for module_name in os.listdir(plugin_dir):
        if file_re.match(module_name):
            log.debug("Examining file: %s", module_name)

            module_name = osp.splitext(module_name)[0]
            local_dir = osp.relpath(plugin_dir, osp.dirname(__file__))
            if '/' in local_dir:
                raise ValueError("Plugin directory not located in local module "
                                 "space! Given: '%s'" % plugin_dir)

            module_path = '.'.join([__name__, local_dir, module_name])
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
                elif issubclass(fd_classes, FeatureDescriptor):
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
                if issubclass(fd_classes, FeatureDescriptor):
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


def get_image_descriptors():
    """
    :return: The map of available image feature descriptors
    :rtype: dict of (str, FeatureDescriptor)
    """
    return _get_plugins(osp.join(osp.dirname(__file__), "image"))


def get_video_descriptors():
    """
    :return: The map of available video feature descriptors
    :rtype: dict of (str, FeatureDescriptor)
    """
    return _get_plugins(osp.join(osp.dirname(__file__), "video"))
