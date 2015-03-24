"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import multiprocessing
import numpy
import os
import os.path as osp
import re
import traceback

from SMQTK.utils import safe_create_dir


def _async_feature_generator_helper(data, descriptor):
    """

    :param data: Data to generate feature over
    :type data: DataFile

    :param descriptor: Feature descriptor that will generate the feature
    :type descriptor: SMQTK.FeatureDescriptors.FeatureDescriptor

    :return: UID and associated feature vector
    :rtype: (int, numpy.core.multiarray.ndarray)

    """
    log = logging.getLogger("_async_feature_generator_helper")
    try:
        log.debug("Generating feature for [%s] -> %s", data, data.filepath)
        feat = descriptor.compute_feature(data)
        # Invalid feature matrix if there are inf or NaN values
        # noinspection PyUnresolvedReferences
        if numpy.isnan(feat.sum()):
            log.error("[%s] Computed feature has NaN values.", data)
            return None
        return feat
    except Exception, ex:
        log.error("[%s] Failed feature generation\n"
                  "Error: %s\n"
                  "Traceback:\n"
                  "%s",
                  data, str(ex), traceback.format_exc())
        return None


class FeatureDescriptor (object):
    """
    Base abstract Feature Descriptor interface
    """
    __metaclass__ = abc.ABCMeta

    # TODO: Input data type white-list + black-lis?
    #       - Requires data objects to specify what data type they are.
    #       - function for telling a user whether it will accept a data element
    #         or not. Ideally a static/class method.

    # Number of cores to use when doing parallel multiprocessing operations
    # - None means use all available cores.
    PARALLEL = None

    def __init__(self, data_directory, work_directory):
        """
        Initialize a feature descriptor instance

        :param data_directory: Feature descriptor data directory.
        :type data_directory: str

        :param work_directory: Work directory for this feature descriptor to use.
        :type work_directory: str

        """
        # Directory that permanent data for this feature descriptor will be
        # held, if any
        self._data_dir = data_directory
        # Directory that work for this feature descriptor should be put. This
        # should be considered a temporary
        self._work_dir = work_directory

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
            safe_create_dir(self._data_dir)
        return self._data_dir

    @property
    def work_directory(self):
        """
        :return: Work directory for this feature descriptor
        :rtype: str
        """
        if not os.path.isdir(self._work_dir):
            safe_create_dir(self._work_dir)
        return self._work_dir

    @abc.abstractproperty
    def has_model(self):
        """
        :return: True if this FeatureDescriptor instance has a valid mode, and
            False if it doesn't.
        :rtype: bool
        """
        return

    @abc.abstractmethod
    def generate_model(self, data_list, parallel=None, **kwargs):
        """
        Generate this feature detector's data-model given a file ingest. This
        saves the generated model to the currently configured data directory.

        This method emits a warning message but does nothing if there is already
        a model generated.

        :param data_list: List of input data elements to generate model with.
        :type data_list: list of SMQTK.utils.DataFile.DataFile
            or tuple of SMQTK.utils.DataFile.DataFile

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        """
        return

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

    def compute_feature_async(self, *data, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        This function does not use the class attribute PARALLEL for determining
        parallel factor as this method can take that specification as an
        argument.

        :param data: List of data elements to compute features for. These must
            have UIDs assigned for feature association in return value
        :type data: list of SMQTK.utils.DataFile.DataFile

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        :param pool_type: multiprocessing pool type to use. If no provided, we
            use a normal multiprocessing.pool.Pool instance.
        :type pool_type: type

        :return: Mapping of data UID to computed feature vector
        :rtype: dict of (int, numpy.core.multiarray.ndarray)

        """
        # Make sure that all input data have associated UIDs
        for item in data:
            # Make sure data items have valid UIDs
            if item.uid is None:
                raise RuntimeError("Some data elements do not have UIDs "
                                   "assigned to them.")
            # args.append((item.uid, item, self))
        self.log.info("Async compute features processing %d elements",
                      len(data))

        self.log.debug("starting pool...")
        parallel = kwds.get('parallel', None)
        pool_t = kwds.get('pool_type', multiprocessing.Pool)
        pool = pool_t(processes=parallel)
        #: :type: dict of (int, multiprocessing.pool.ApplyResult)
        ar_map = {}
        for d in data:
            ar_map[d.uid] = pool.apply_async(_async_feature_generator_helper,
                                             args=(d, self))

        #: :type: dict of (int, numpy.core.multiarray.ndarray)
        r_dict = {}
        failures = False
        for i, (uid, ar) in enumerate(ar_map.iteritems()):
            feat = ar.get()
            if feat is None:
                failures = True
                continue
            else:
                r_dict[uid] = feat
            self.log.info("Progress: [%d/%d] %3.3f%%",
                          i+1, len(ar_map), float(i+1)/(len(ar_map)) * 100)
        # Check for failed generation
        if failures:
            raise RuntimeError("Failure occurred during data feature "
                               "computation. See logging.")

        pool.close()
        pool.join()

        return r_dict


def get_descriptors():
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
    :rtype: dict of (str, type)

    """
    log = logging.getLogger("get_descriptors")
    class_map = {}

    this_dir = os.path.abspath(os.path.dirname(__file__))
    log.debug("Searching in directory: %s", this_dir)

    file_re = re.compile("^[a-zA-Z].*(?:\.py)?$")
    standard_var = "FEATURE_DESCRIPTOR_CLASS"

    for module_name in os.listdir(this_dir):
        if file_re.match(module_name):
            log.debug("Examining file: %s", module_name)

            module_name = osp.splitext(module_name)[0]

            module_path = '.'.join([__name__, module_name])
            log.debug("Attempting import of: %s", module_path)
            module = __import__(module_path, fromlist=__name__)

            # Look for standard variable
            fd_classes = []
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
