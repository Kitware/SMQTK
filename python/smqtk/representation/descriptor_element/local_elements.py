import numpy
import os.path as osp
import threading

from smqtk.representation import DescriptorElement
from smqtk.utils import file_utils
from smqtk.utils.string_utils import partition_string


__author__ = "paul.tunison@kitware.com"


class DescriptorMemoryElement (DescriptorElement):
    """
    In-memory representation of descriptor elements. Stored vectors are
    effectively immutable.
    """

    # In-memory cache of descriptor vectors
    MEMORY_CACHE = {}
    MEMORY_CACHE_LOCK = threading.RLock()

    @classmethod
    def is_usable(cls):
        return True

    def __getstate__(self):
        return self.type(), self.uuid(), self.vector(),

    def __setstate__(self, state):
        self._type_label = state[0]
        self._uuid = state[1]
        with self.MEMORY_CACHE_LOCK:
            self.MEMORY_CACHE[self._get_cache_index()] = state[2]

    def _get_cache_index(self):
        """
        :return: Index tuple for this element in the global cache
        :rtype: (str, collections.Hashable)
        """
        return self.type(), self.uuid()

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
        with self.MEMORY_CACHE_LOCK:
            return self._get_cache_index() in self.MEMORY_CACHE

    def vector(self):
        """
        Implementation Note
        -------------------
        A copy of the internally stored vector is returned in order to mimic
        immutability.

        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None

        """
        i = self._get_cache_index()
        with self.MEMORY_CACHE_LOCK:
            if i in self.MEMORY_CACHE:
                return numpy.copy(self.MEMORY_CACHE[i])
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
        :type new_vec: numpy.core.multiarray.ndarray | None

        """
        idx = self._get_cache_index()
        with self.MEMORY_CACHE_LOCK:
            if new_vec is None and idx in self.MEMORY_CACHE:
                del self.MEMORY_CACHE[self._get_cache_index()]
            else:
                self.MEMORY_CACHE[idx] = numpy.copy(new_vec)


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
        :type uuid: collections.Hashable

        :param save_dir: Directory to save this element's contents. If this path
            is relative, we interpret as relative to the current working
            directory.
        :type save_dir: str | unicode

        :param subdir_split: If a positive integer, this will cause us to store
            the vector file in a subdirectory under the ``save_dir`` that was
            specified. The integer value specifies the number of splits that we
            will make in the stringification of this descriptor's UUID. If there
            happen to be dashes in this stringification, we will remove them
            (as would happen if given an uuid.UUID instance as the uuid
            element).
        :type subdir_split: None | int

        """
        super(DescriptorFileElement, self).__init__(type_str, uuid)

        self._save_dir = osp.abspath(osp.expanduser(save_dir))

        # Saving components
        self._subdir_split = subdir_split
        if subdir_split and int(subdir_split) > 0:
            save_dir = osp.join(self._save_dir,
                                *partition_string(str(uuid).replace('-', ''),
                                                  int(subdir_split))
                                )
        else:
            save_dir = self._save_dir

        self._vec_filepath = osp.join(save_dir,
                                      "%s.%s.vector.npy" % (self._type_label,
                                                            str(uuid)))

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

        """
        file_utils.safe_create_dir(osp.dirname(self._vec_filepath))
        numpy.save(self._vec_filepath, new_vec)


DESCRIPTOR_ELEMENT_CLASS = [
    DescriptorMemoryElement,
    DescriptorFileElement,
]
