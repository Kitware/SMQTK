import abc
import numpy

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.utils import ContentTypeValidator
from smqtk.utils.parallel import parallel_map


DFLT_DESCRIPTOR_FACTORY = DescriptorElementFactory(DescriptorMemoryElement, {})


class DescriptorGenerator (SmqtkAlgorithm, ContentTypeValidator):
    """
    Base abstract Feature Descriptor interface
    """

    def compute_descriptor(self, data, descr_factory=DFLT_DESCRIPTOR_FACTORY,
                           overwrite=False):
        """
        Given some data, return a descriptor element containing a descriptor
        vector.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.representation.DataElement

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance. The default factory produces
            ``DescriptorMemoryElement`` instances by default.
        :type descr_factory: smqtk.representation.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vector for the given data even when there exists a precomputed
            vector in the generated DescriptorElement as generated from the
            provided factory. This will overwrite the persistently stored vector
            if the provided factory produces a DescriptorElement implementation
            with such storage.
        :type overwrite: bool

        :return: Result descriptor element. UUID of this output descriptor is
            the same as the UUID of the input data element.
        :rtype: smqtk.representation.DescriptorElement

        """
        # Check content type against listed valid types
        ct = data.content_type()
        if ct not in self.valid_content_types():
            self._log.error("Cannot compute descriptor from content type '%s' "
                            "data: %s)" % (ct, data))
            raise ValueError("Cannot compute descriptor from content type '%s' "
                             "data: %s)" % (ct, data))

        # Produce the descriptor element container via the provided factory
        # - If the generated element already contains a vector, because the
        #   implementation provides some kind of persistent caching mechanism
        #   or something, don't compute another descriptor vector unless the
        #   overwrite flag is True
        descr_elem = descr_factory.new_descriptor(self.name, data.uuid())
        if overwrite or not descr_elem.has_vector():
            vec = self._compute_descriptor(data)
            descr_elem.set_vector(vec)
        else:
            self._log.debug("Found existing vector in generated element: %s",
                            descr_elem)

        return descr_elem

    def compute_descriptor_async(self, data_iter,
                                 descr_factory=DFLT_DESCRIPTOR_FACTORY,
                                 overwrite=False, procs=None, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        Base implementation additional keyword arguments:
            use_mp [= False]
                If multi-processing should be used vs. multi-threading.

        :param data_iter: Iterable of data elements to compute features for.
            These must have UIDs assigned for feature association in return
            value.
        :type data_iter: collections.Iterable[smqtk.representation.DataElement]

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance. The default factory produces
            ``DescriptorMemoryElement`` instances by default.
        :type descr_factory: smqtk.representation.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vectors for the given data even when there exists precomputed
            vectors in the generated DescriptorElements as generated from the
            provided factory. This will overwrite the persistently stored
            vectors if the provided factory produces a DescriptorElement
            implementation such storage.
        :type overwrite: bool

        :param procs: Optional specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type procs: int | None

        :raises ValueError: An input DataElement was of a content type that we
            cannot handle.

        :return: Mapping of input DataElement UUIDs to the computed descriptor
            element for that data. DescriptorElement UUID's are congruent with
            the UUID of the data element it is the descriptor of.
        :rtype: dict[collections.Hashable,
                     smqtk.representation.DescriptorElement]

        """
        self._log.info("Async compute features")

        use_mp = kwds.get('use_mp', False)

        def work(d):
            return d, self.compute_descriptor(d, descr_factory, overwrite)

        results = parallel_map(work, data_iter, cores=procs, ordered=False,
                               use_multiprocessing=use_mp)

        de_map = {}
        for data, descriptor in results:
            de_map[data.uuid()] = descriptor

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
        :type data: smqtk.representation.DataElement

        :return: Feature vector.
        :rtype: numpy.core.multiarray.ndarray

        """
