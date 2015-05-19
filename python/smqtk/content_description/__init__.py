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
import traceback

from smqtk.utils import SimpleTimer


def _async_feature_generator_helper(data, descriptor):
    """

    :param data: Data to generate feature over
    :type data: DataFile

    :param descriptor: Feature descriptor that will generate the feature
    :type descriptor: SMQTK.content_description.ContentDescriptor

    :return: UID and associated feature vector
    :rtype: (int, numpy.core.multiarray.ndarray)

    """
    log = logging.getLogger("_async_feature_generator_helper")
    try:
        # log.debug("Generating feature for [%s] -> %s", data, data.filepath)
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


class ContentDescriptor (object):
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

    @abc.abstractmethod
    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return

    @abc.abstractmethod
    def generate_model(self, data_set, **kwargs):
        """
        Generate this feature detector's data-model given a file ingest. This
        saves the generated model to the currently configured data directory.

        This method does nothing if there is already a model generated or if
        this descriptor does not generate a model.

        This abstract super method should be invoked for common error checking.

        :raises ValueError: One or more input data elements did not conform to
            this descriptor's valid content set.

        :param data_set: Set of input data elements to generate the model
            with.
        :type data_set: collections.Set[smqtk.data_rep.DataElement]

        """
        valid_types = self.valid_content_types()
        invalid_types_found = set()
        for di in data_set:
            if di.content_type() not in valid_types:
                invalid_types_found.add(di.content_type())
        if invalid_types_found:
            self.log.error("Found one or more invalid content types among "
                           "input:")
            for t in sorted(invalid_types_found):
                self.log.error("\t- '%s", t)
            raise ValueError("Discovered invalid content type among input "
                             "data: %s" % sorted(invalid_types_found))

    @abc.abstractmethod
    def compute_feature(self, data):
        """
        Given some kind of data, process and return a feature vector as a Numpy
        array.

        This abstract super method should be invoked for common error checking.

        :raises RuntimeError: Feature extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.data_rep.DataElement

        :return: Feature vector.
        :rtype: numpy.ndarray

        """
        ct = data.content_type()
        if ct not in self.valid_content_types():
            self.log.error("Cannot compute descriptor of content type '%s'", ct)
            raise ValueError("Cannot compute descriptor of content type '%s'"
                             % ct)

    def compute_feature_async(self, *data, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        This function does not use the class attribute PARALLEL for determining
        parallel factor as this method can take that specification as an
        argument.

        :param data: List of data elements to compute features for. These must
            have UIDs assigned for feature association in return value
        :type data: list[smqtk.data_rep.DataElement]

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
        self.log.info("Async compute features processing %d elements",
                      len(data))

        parallel = kwds.get('parallel', None)
        pool_t = kwds.get('pool_type', multiprocessing.Pool)
        pool = pool_t(processes=parallel)
        #: :type: dict of (int, multiprocessing.pool.ApplyResult)
        ar_map = {}
        with SimpleTimer("Starting pool...", self.log.debug):
            for d in data:
                ar_map[d.uuid()] = \
                    pool.apply_async(_async_feature_generator_helper,
                                     args=(d, self))
        pool.close()

        #: :type: dict[int, numpy.core.multiarray.ndarray]
        r_dict = {}
        failures = False
        # noinspection PyPep8Naming
        perc_T = 0.0
        perc_inc = 0.1
        with SimpleTimer("Collecting async results...", self.log.debug):
            for i, (uid, ar) in enumerate(ar_map.iteritems()):
                feat = ar.get()
                if feat is None:
                    failures = True
                    continue
                else:
                    r_dict[uid] = feat

                perc = float(i+1)/len(ar_map)
                if perc >= perc_T:
                    self.log.debug("Progress: [%d/%d] %3.3f%%",
                                   i+1, len(ar_map),
                                   float(i+1)/(len(ar_map)) * 100)
                    perc_T += perc_inc
        pool.join()

        # Check for failed generation
        if failures:
            raise RuntimeError("Failure occurred during data feature "
                               "computation. See logging.")

        return r_dict


def get_descriptors():
    """
    Discover and return ContentDescriptor classes found in the given plugin
    search directory. Keys in the returned map are the names of the discovered
    classes, and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``CONTENT_DESCRIPTOR_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same name and casing as the module. If
    neither are found, the module is skipped.

    :return: Map of discovered class object of type ``ContentDescriptor`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "CONTENT_DESCRIPTOR_CLASS"
    return get_plugins(__name__, this_dir, helper_var, ContentDescriptor)
