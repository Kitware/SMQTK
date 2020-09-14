import numpy
import os.path as osp

from six import BytesIO

from smqtk.representation import DescriptorElement
from smqtk.utils.file import safe_create_dir
from smqtk.utils.string import partition_string


class DescriptorMemoryElement (DescriptorElement):
    """
    In-memory representation of descriptor elements. Stored vectors are
    effectively immutable.

    Example
    -------
    >>> self = DescriptorMemoryElement('random', 0)
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_str, uuid):
        super(DescriptorMemoryElement, self).__init__(type_str, uuid)
        self.__v = None

    def __getstate__(self):
        state = super(DescriptorMemoryElement, self).__getstate__()
        # save vector as binary string
        b = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(b, self.vector())
        state['v'] = b.getvalue()
        return state

    def __setstate__(self, state):
        # Handle the previous state format:
        if isinstance(state, tuple):
            self._type_label = state[0]
            self._uuid = state[1]
            b = BytesIO(state[2])
        else:  # dictionary
            super(DescriptorMemoryElement, self).__setstate__(state)
            b = BytesIO(state['v'])
        self.__v = numpy.load(b)

    def get_config(self):
        """
        :return: JSON type compliant configuration dictionary.
        :rtype: dict
        """
        return {}

    def has_vector(self):
        """
        :return: Whether or not this container current has a descriptor vector
            stored.
        :rtype: bool
        """
        return self.__v is not None

    def vector(self):
        """
        Implementation Note
        -------------------
        A copy of the internally stored vector is returned.

        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None

        """
        # Copy if storing an array, otherwise return the None value
        if self.__v is not None:
            return numpy.copy(self.__v)
        return None

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        ``new_vec`` may be None, which clears this descriptor's vector from the
        cache.

        Implementation Note
        -------------------
        This implementation copies input arrays before storage to mimic
        immutability.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray | tuple | list | None

        :returns: Self.
        :rtype: DescriptorMemoryElement

        """
        # Copy a non-None value given, otherwise stay None
        if new_vec is not None:
            self.__v = numpy.copy(new_vec)
        else:
            self.__v = None
        return self


class DescriptorFileElement (DescriptorElement):
    """
    File-based storage of descriptor element.

    When initialized, saves uuid and vector as a serialized pickle-file and
    numpy-format npy file, respectively. These are named according to the string
    representation of the uuid object provided. These are then loaded from disk
    when the ``uuid`` or ``vector`` methods are called. This is in turn slower
    performance wise than ``MemoryElement``, however RAM consumption will be
    lower for large number of elements that would otherwise exceed RAM storage
    space.

    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_str, uuid, save_dir, subdir_split=None):
        """
        Initialize a file-base descriptor element.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: uuid for this descriptor
        :type uuid: collections.abc.Hashable

        :param save_dir: Directory to save this element's contents. If this path
            is relative, we interpret as relative to the current working
            directory.
        :type save_dir: str | unicode

        :param subdir_split: If a positive integer and greater than 1, this will
            cause us to store the vector file in a subdirectory under the
            ``save_dir`` based on our ``uuid``. The integer value specifies the
            number of splits that we will make in the stringification of this
            descriptor's UUID. The last split component is left off when
            determining the save directory (thus the >1 above).

            Dashes are stripped from this string (as would happen if given an
            uuid.UUID instance as the uuid element).
        :type subdir_split: None | int

        """
        super(DescriptorFileElement, self).__init__(type_str, uuid)
        self._save_dir = osp.abspath(osp.expanduser(save_dir))
        self._subdir_split = subdir_split

        # Generate filepath from parameters
        if self._subdir_split and int(self._subdir_split) > 1:
            # TODO: If uuid is an integer, create string with left-padded 0's
            #       to expand out the "length" before partitioning.
            save_dir = osp.join(
                self._save_dir,
                *partition_string(str(self.uuid()).replace('-', ''),
                                  int(self._subdir_split))[:-1]
            )
        else:
            save_dir = self._save_dir
        self._vec_filepath = osp.join(save_dir,
                                      "%s.%s.vector.npy" % (self.type(),
                                                            str(self.uuid())))

    def __getstate__(self):
        state = super(DescriptorFileElement, self).__getstate__()
        state.update({
            '_save_dir': self._save_dir,
            '_subdir_split': self._subdir_split,
            '_vec_filepath': self._vec_filepath,
        })
        return state

    def __setstate__(self, state):
        super(DescriptorFileElement, self).__setstate__(state)
        self._save_dir = state['_save_dir']
        self._subdir_split = state['_subdir_split']
        self._vec_filepath = state['_vec_filepath']

    def get_config(self):
        return {
            "save_dir": self._save_dir,
            'subdir_split': self._subdir_split
        }

    def has_vector(self):
        """
        :return: Whether or not this container current has a descriptor vector
            stored.
        :rtype: bool
        """
        return osp.isfile(self._vec_filepath)

    def vector(self):
        """
        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None
        """
        # TODO: Load as memmap?
        #       i.e. modifications by user to vector will be reflected on disk.
        if self.has_vector():
            return numpy.load(self._vec_filepath)
        else:
            return None

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray

        :returns: Self.
        :rtype: DescriptorFileElement

        """
        safe_create_dir(osp.dirname(self._vec_filepath))
        numpy.save(self._vec_filepath, new_vec)
        return self


DESCRIPTOR_ELEMENT_CLASS = [
    DescriptorMemoryElement,
    DescriptorFileElement,
]
