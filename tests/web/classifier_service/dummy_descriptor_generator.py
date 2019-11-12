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
        return ['text/plain']

    def _compute_descriptor(self, data):
        """
        Internal method that defines the generation of the descriptor vector
        for a given data element. This returns a numpy array.

        This method is only called if the data element has been verified to be
        of a valid content type for this descriptor implementation.

        :raises RuntimeError: Feature extraction failure of some kind.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.representation.DataElement

        :return: Feature vector.
        :rtype: numpy.core.multiarray.ndarray

        """
        return np.zeros((5,), np.float64)
