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
from smqtk.utils.configurable_interface import Configurable


class ContentDescriptor (Configurable):
    """
    Base abstract Feature Descriptor interface
    """
    __metaclass__ = abc.ABCMeta

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

    # noinspection PyMethodParameters
    @abc.abstractmethod
    def is_usable(cls):
        """
        Check whether this descriptor is available for use.

        Since certain ContentDescriptor implementations may require additional
        dependencies that may not yet be available on the system, this method
        should check for those dependencies and return a boolean saying if the
        implementation is usable.

        NOTES:
            - This should be a class method
            - When not available, this should emit a warning message pointing to
                documentation on how to get/install required dependencies.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        return

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

        This method should do nothing if there is already a model generated or
        if this descriptor does not generate a model.

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

    def compute_descriptor(self, data, descr_factory, overwrite=False):
        """
        Given some kind of data, return a descriptor element containing a
        descriptor vector.

        This abstract super method should be invoked for common error checking.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.data_rep.DataElement

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance.
        :type descr_factory: smqtk.data_rep.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vector for the given data even when there exists a precomputed
            vector in the generated DescriptorElement as generated from the
            provided factory. This will overwrite the persistently stored vector
            if the provided factory produces a DescriptorElement implementation
            with such storage.
        :type overwrite: bool

        :return: Result descriptor element. UUID of this output descriptor is
            the same as the UUID of the input data element.
        :rtype: smqtk.data_rep.DescriptorElement

        """
        # Check content type against listed valid types
        ct = data.content_type()
        if ct not in self.valid_content_types():
            self.log.error("Cannot compute descriptor of content type '%s'", ct)
            raise ValueError("Cannot compute descriptor of content type '%s'"
                             % ct)

        # Produce the descriptor element container via the provided factory
        # - If the generated element already contains a vector, because the
        #   implementation provides some kind of persistent caching mechanism or
        #   something, don't compute another descriptor vector unless the
        #   overwrite flag is True
        descr_elem = descr_factory.new_descriptor(self.name, data.uuid())
        if overwrite or not descr_elem.has_vector():
            vec = self._compute_descriptor(data)
            descr_elem.set_vector(vec)
        else:
            self.log.debug("Found existing vector in generated element.")

        return descr_elem

    def compute_descriptor_async(self, data_iter, descr_factory,
                                 overwrite=False, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        This function does NOT use the class attribute PARALLEL for determining
        parallel factor as this method can take that specification as an
        argument.

        :param data_iter: Iterable of data elements to compute features for.
            These must have UIDs assigned for feature association in return
            value.
        :type data_iter: collections.Iterable[smqtk.data_rep.DataElement]

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instances.
        :type descr_factory: smqtk.data_rep.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vectors for the given data even when there exists precomputed
            vectors in the generated DescriptorElements as generated from the
            provided factory. This will overwrite the persistently stored
            vectors if the provided factory produces a DescriptorElement
            implementation such storage.
        :type overwrite: bool

        :param parallel: Optionally specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type parallel: int

        :param pool_type: multiprocessing pool type to use. If no provided, we
            use a normal multiprocessing.pool.Pool instance.
        :type pool_type: type

        :return: Mapping of data element UUID to computed descriptor element.
            Element UUID's are congruent with the UUID of the data element it
            is the descriptor of.
        :rtype: dict[collections.Hashable, smqtk.data_rep.DescriptorElement]

        """
        self.log.info("Async compute features")

        # Mapping of DataElement UUID to async processing result
        #: :type: dict[collections.Hashable, multiprocessing.pool.ApplyResult]
        ar_map = {}
        # Mapping of DataElement UUID to the DescriptorElement for it.
        #: :type: dict[collections.Hashable, smqtk.data_rep.DescriptorElement]
        de_map = {}

        # Queue up descriptor generation for descriptor elements that
        parallel = kwds.get('parallel', None)
        pool_t = kwds.get('pool_type', multiprocessing.Pool)
        pool = pool_t(processes=parallel)
        with SimpleTimer("Queuing descriptor computation...", self.log.debug):
            for d in data_iter:
                de_map[d.uuid()] = descr_factory.new_descriptor(self.name,
                                                                d.uuid())
                if overwrite or not de_map[d.uuid()].has_vector():
                    ar_map[d.uuid()] = \
                        pool.apply_async(_async_feature_generator_helper,
                                         args=(self, d))
        pool.close()

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
                    de_map[uid].set_vector(feat)

                perc = float(i + 1) / len(ar_map)
                if perc >= perc_T:
                    self.log.debug("Progress: [%d/%d] %3.3f%%",
                                   i + 1, len(ar_map),
                                   float(i + 1) / (len(ar_map)) * 100)
                    perc_T += perc_inc
        pool.join()

        # Check for failed generation
        if failures:
            raise RuntimeError("Failure occurred during data feature "
                               "computation. See logging.")

        return de_map

    @abc.abstractmethod
    def _compute_descriptor(self, data):
        """
        Internal method that defines the generation of the descriptor vector for
        a given data element. This returns a numpy array.

        This method is only called if the data element has been verified to be
        of a valid content type for this descriptor implementation.

        :raises RuntimeError: Feature extraction failure of some kind.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.data_rep.DataElement

        :return: Feature vector.
        :rtype: numpy.core.multiarray.ndarray

        """
        return


def _async_feature_generator_helper(cd_inst, data):
    """
    Helper method for asynchronously producing a descriptor vector.

    :param data: Data to generate feature over
    :type data: smqtk.data_rep.DataElement

    :param cd_inst: Content descriptor that will generate the feature
    :type cd_inst: SMQTK.content_description.ContentDescriptor

    :return: UID and associated feature vector
    :rtype: numpy.core.multiarray.ndarray or None
    """
    log = logging.getLogger("_async_feature_generator_helper")
    try:
        # noinspection PyProtectedMember
        feat = cd_inst._compute_descriptor(data)
        # Invalid feature matrix if there are inf or NaN values
        # noinspection PyUnresolvedReferences
        if numpy.isnan(feat.sum()):
            log.error("[%s] Computed feature has NaN values.", data)
            return None
        elif float('inf') in feat:
            log.error("[%s] Computed feature has infinite values", data)
            return None
        return feat
    except Exception, ex:
        log.error("[%s] Failed feature generation\n"
                  "Error: %s\n"
                  "Traceback:\n"
                  "%s",
                  data, str(ex), traceback.format_exc())
        return None


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
    return get_plugins(__name__, this_dir, helper_var, ContentDescriptor,
                       lambda c: c.is_usable())
