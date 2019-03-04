import hashlib

import numpy
from six import BytesIO

from smqtk.exceptions import ReadOnlyError
from smqtk.representation import DataElement


class MatrixDataElement (DataElement):
    """
    DataElement whose data is represented in memory by a ``numpy.ndarray``
    instance.

    This implementation additionally provides a ``matrix`` property that exposes
    a natively stored ``numpy.ndarray`` (may be None).  It is expected that this
    implementation is to be used with components that have ``matrix`` short-cuts
    or otherwise where the matrix is the important that data is accessed often.

    Since the ndarray is stored natively, ``get_bytes()`` bytes are generated on
    the fly based on the current state of matrix.  The ``writable()`` method on
    this instance only pertains to both the ``set_bytes()`` method AND
    ``matrix`` property setter.
    """

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional dependencies
        that may not yet be available on the system, this method should check
        for those dependencies and return a boolean saying if the implementation
        is usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should emit a
                warning detailing why the implementation is not available for
                use.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        return True

    def __init__(self, mat=None, readonly=False):
        """
        :param None|collections.Sequence|numpy.ndarray mat:
            Optional matrix to store at construction time.
        :param bool readonly:
            If the matrix stored should be considered read-only. This pertains
            to both the ``set_bytes`` method AND setting to the ``matrix``
            property.  This does NOT pertain to modifying an already set matrix,
            which should be controlled by setting flags on the ndarray instance.
        """
        super(MatrixDataElement, self).__init__()
        self._matrix = None
        if mat is not None:
            self._matrix = numpy.asarray(mat)
        self._readonly = bool(readonly)

    def __repr__(self):
        return super(DataElement, self).__repr__() + \
            "{{shape: {}}}" \
            .format(self._matrix.shape,)

    @property
    def matrix(self):
        """
        :return: Get the matrix stored in this element. This may be None if
            there is no matrix currently stored in this element (is empty).
            Alternatively, the matrix may be an "empty" shape, or have zero
            area.
        :rtype: None | numpy.ndarray
        """
        return self._matrix

    @matrix.setter
    def matrix(self, m):
        """
        :param numpy.ndarray m:
            New ndarray instance to set as the contained matrix.

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.
        """
        # Invoking super ``set_bytes`` for common ReadOnlyError functionality
        # based on ``writable`` method.
        super(MatrixDataElement, self).set_bytes(b'')
        self._matrix = numpy.asarray(m)

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        mat_json = None
        if self._matrix is not None:
            mat_json = self._matrix.tolist()
        return {
            'mat': mat_json,
            'readonly': self._readonly,
        }

    def content_type(self):
        """
        :return: Standard type/subtype string for this data element, or None if
            the content type is unknown.
        :rtype: str or None
        """
        # Blob of bytes (numpy save format)
        return 'application/octet-stream'

    def is_empty(self):
        """
        Check if this element contains no bytes.

        The intend of this method is to quickly check if there is any data
        behind this element, ideally without having to read all/any of the
        underlying data.

        :return: If this element contains 0 bytes.
        :rtype: bool

        """
        return self._matrix is None or self._matrix.size == 0

    def get_bytes(self):
        """
        :return: Get the bytes for this data element.
        :rtype: bytes
        """
        if self._matrix is not None:
            buf = BytesIO()
            # noinspection PyTypeChecker
            numpy.save(buf, self._matrix)
            return buf.getvalue()
        else:
            return bytes()

    def writable(self):
        """
        :return: if this instance supports setting bytes.
        :rtype: bool
        """
        return not self._readonly

    def set_bytes(self, b):
        """
        Set bytes to this data element.

        Not all implementations may support setting bytes (check ``writable``
        method return).

        :param b: bytes to set.
        :type b: bytes

        :raises ReadOnlyError: This data element can only be read from / does
            not support writing.

        """
        super(MatrixDataElement, self).set_bytes(b)
        if b:
            buf = BytesIO(b)
            self._matrix = numpy.load(buf)
        else:
            self._matrix = None


