__author__ = 'purg'

import copy
import numpy
import os
import os.path as osp

from smqtk.data_rep import DescriptorElement
from smqtk.utils import safe_create_dir
from smqtk.utils.string_utils import partition_string
from smqtk_config import WORK_DIR


class DescriptorMemoryElement (DescriptorElement):
    """
    In-memory representation of descriptor elements.
    """

    # In-memory cache of descriptor vectors
    MEMORY_CACHE = {}

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
        return self.uuid() in DescriptorMemoryElement.MEMORY_CACHE

    def vector(self):
        """
        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None
        """
        return DescriptorMemoryElement.MEMORY_CACHE.get(self.uuid(), None)

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray

        """
        DescriptorMemoryElement.MEMORY_CACHE[self.uuid()] = new_vec


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

    def __init__(self, type_str, uuid, save_dir, work_relative=False,
                 subdir_split=None):
        """
        Initialize a file-base descriptor element.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: uuid for this descriptor
        :type uuid: collections.Hashable

        :param save_dir: Directory to save this element's contents. If this path
            is relative, we interpret as relative to the WORK_DIR path set in
            the `smqtk_config` module.
        :type save_dir: str | unicode

        :param work_relative: If true, we should interpret ``root_directory`` as
            relative to the configured WORK_DIR parameter in the
            ``smqtk_config`` module.
        :type work_relative: bool

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

        self._save_dir = osp.abspath(
            osp.join(
                WORK_DIR if work_relative else os.getcwd(),
                osp.expanduser(save_dir)
            )
        )

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
            "work_relative": False,
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
        :return: The descriptor vector as a numpy array.
        :rtype: numpy.core.multiarray.ndarray
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
        safe_create_dir(osp.dirname(self._vec_filepath))
        numpy.save(self._vec_filepath, new_vec)


DESCRIPTOR_ELEMENT_CLASS = [
    DescriptorMemoryElement,
    DescriptorFileElement,
]
