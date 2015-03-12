# coding=utf-8

import abc
import logging
import os
import os.path as osp


class FeatureDescriptor (object):
    """
    base class for feature generation methods
    """
    __metaclass__ = abc.ABCMeta

    BACKGROUND_RATIO = 0.33

    def __init__(self, base_data_dir, base_work_dir):
        """ Basic construction

        :param base_data_dir: Root data directory
        :type base_data_dir: str
        :param base_work_dir: Root work directory
        :type base_work_dir: str

        """
        self._data_dir = osp.join(osp.abspath(osp.expanduser(base_data_dir)),
                                  "features")
        self._work_dir = osp.join(osp.abspath(osp.expanduser(base_work_dir)),
                                  "features")

        if not osp.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        if not osp.isdir(self._work_dir):
            os.makedirs(self._work_dir)

    @property
    def _log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    #
    # Abstract methods
    #

    @abc.abstractproperty
    def name(self):
        """
        :return: Name of the descriptor
        :rtype: str
        """
        return

    @abc.abstractproperty
    def ids_file(self):
        """
        :return: index-to-ID mapping file path
        :rtype: str
        """
        return

    @abc.abstractproperty
    def bg_flags_file(self):
        """
        :return: index-to-background flag file path
        :rtype: str
        """
        return

    @abc.abstractproperty
    def feature_data_file(self):
        """
        :return: feature data file path
        :rtype: str
        """
        return

    @abc.abstractproperty
    def kernel_data_file(self):
        """
        :return: distance kernel data file path
        :rtype: str
        """
        return

    @abc.abstractmethod
    def image_feature(self, image_file):
        """
        Process the given image file, returning a numpy array that is the
        image's feature vector.

        :param image_file: Path to the image file to generate a feature vector
        :type image_file: str

        :return: Image feature vector
        :rtype: numpy.array

        """
        return

    @abc.abstractmethod
    def generate_feature_data(self, ingest_manager, **kwds):
        """
        Create feature data over the given ingest.

        This should probably only be called by tools that don't mind blocking
        for many hours.

        Generated files include:
            - ordered index-to-ID list (effective mapping)
            - ordered background flags list (parallel to ID list/mapping)
            - Feature matrix file
            - Distance kernel file

        :param ingest_manager: The ingest to create data files over.
        :type ingest_manager: IngestManager

        """
        return