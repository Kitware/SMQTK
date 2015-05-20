__author__ = 'purg'

import cPickle
import numpy
import os.path as osp

from smqtk.data_rep import DescriptorElement


class MemoryElement (DescriptorElement):
    """
    In-memory representation of descriptor elements.
    """

    def __init__(self, uuid, vector):
        """
        Initialize in-memory descriptor element.

        :param uuid: Unique ID value for this vector
        :type uuid: collections.Hashable

        :param vector: Numpy ndarray to store
        :type vector: numpy.core.multiarray.ndarray

        """
        self._uuid = uuid
        self._vector = vector

    def uuid(self):
        """
        :return: Unique ID for this vector.
        :rtype: collections.Hashable
        """
        return self._uuid

    def vector(self):
        """
        :return: The descriptor vector as a numpy array.
        :rtype: numpy.core.multiarray.ndarray
        """
        return self._vector


class FileElement (DescriptorElement):
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

    def __init__(self, uuid, vector, save_dir):
        """
        Initialize a file-base descriptor element.

        :param uuid: uuid for this descriptor
        :type uuid: collections.Hashable

        :param vector: Numpy array.
        :type vector: numpy.core.multiarray.ndarray

        :param save_dir: Directory to save this element's contents.
        :type save_dir: str | unicode

        """
        self._save_dir = osp.abspath(osp.expanduser(save_dir))

        # Saving components
        self._uuid_filepath = osp.join(self._save_dir,
                                       "%s.uuid.pickle" % str(uuid))
        self._vec_filepath = osp.join(self._save_dir,
                                      "%s.vactor.npy" % str(uuid))

        numpy.save(self._vec_filepath, vector)
        with open(self._uuid_filepath, 'wb') as ofile:
            cPickle.dump(uuid, ofile)

    def uuid(self):
        """
        :return: Unique ID for this vector.
        :rtype: collections.Hashable
        """
        with open(self._uuid_filepath, 'rb') as uuid_file:
            return cPickle.load(uuid_file)

    def vector(self):
        """
        :return: The descriptor vector as a numpy array.
        :rtype: numpy.core.multiarray.ndarray
        """
        # TODO: Load as memmap?
        #       i.e. modifications by user to vector will be reflected on disk.
        return numpy.load(self._vec_filepath)


DESCRIPTOR_ELEMENT_CLASS = [
    MemoryElement,
    FileElement,
]
