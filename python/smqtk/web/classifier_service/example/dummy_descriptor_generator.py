import numpy as np

from smqtk.algorithms.descriptor_generator import DescriptorGenerator


class DummyDescriptorGenerator (DescriptorGenerator):
    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this
        class's ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        # No constructor, no config
        return {}

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional
        dependencies that may not yet be available on the system, this method
        should check for those dependencies and return a boolean saying if the
        implementation is usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should emit a
                warning detailing why the implementation is not available for
                use.

        :return: Boolean determination of whether this implementation is
                 usable.
        :rtype: bool

        """
        return True

    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return ['image/jpeg']

    def _generate_arrays(self, data_iter):
        """
        Inner template method that defines the generation of descriptor vectors
        for a given iterable of data elements.

        Pre-conditions:
          - Data elements input to this method have been validated to be of at
            least one of this class's reported ``valid_content_types``.

        :param collections.Iterable[DataElement] data_iter:
            Iterable of data element instances to be described.

        :raises RuntimeError: Descriptor extraction failure of some kind.

        :return: Iterable of numpy arrays in parallel association with the
            input data elements.
        :rtype: collections.Iterable[numpy.ndarray]
        """
        for _ in data_iter:
            yield np.zeros((5,), np.float64)
