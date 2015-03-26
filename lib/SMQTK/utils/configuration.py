"""
Helper classes for access to JSON system configuration
"""


import os.path as osp
import smqtk_config

from SMQTK.FeatureDescriptors import get_descriptors
from SMQTK.Indexers import get_indexers
from SMQTK.utils import DataIngest, VideoIngest


class IngestConfiguration (object):
    """
    Interface to system ingest configuration as configured in the file:

        etc/system_config.json

    Provides convenience methods to get the Ingest ingest instance as well
    as factory methods to construct FeatureDescriptor and Indexer instances
    for a given type label.

    """

    INGEST_CONFIG = smqtk_config.SYSTEM_CONFIG['Ingests']

    TYPE_MAP = {
        "image": DataIngest,
        "video": VideoIngest,
    }

    @classmethod
    def available_ingest_labels(cls):
        """
        :return: List of available Ingest labels in no particular order.
        :rtype: list of str
        """
        return cls.INGEST_CONFIG['listing'].keys()

    @classmethod
    def ingest_data_directory(cls):
        """
        :return: The base directory where ingests data should be located
        :rtype: str
        """
        return osp.join(smqtk_config.DATA_DIR, cls.INGEST_CONFIG['dir'])

    @classmethod
    def ingest_work_directory(cls):
        """
        :return: The base directory where ingests work should be located
        :rtype: str
        """
        return osp.join(smqtk_config.WORK_DIR, cls.INGEST_CONFIG['dir'])

    def __init__(self, ingest_label, config_dict=None):
        # Override local base config dict if one was given
        self.INGEST_CONFIG = config_dict or IngestConfiguration.INGEST_CONFIG

        if ingest_label not in self.available_ingest_labels():
            raise ValueError("Given ingest label '%s' not available in "
                             "configuration! Make sure to add configuration "
                             "for an ingest first."
                             % ingest_label)

        label_config = self.INGEST_CONFIG['listing'][ingest_label]

        self.label = ingest_label
        self.data_dir = osp.join(self.ingest_data_directory(), label_config['dir'])
        self.work_dir = osp.join(self.ingest_work_directory(), label_config['dir'])
        self.type = label_config['type']

        self.descriptor_config = label_config['descriptors']
        self.indexer_config = label_config['indexers']

        # Cache of requested FeatureDescriptor instances for sharing with
        # Indexer factory method,
        self._ingest_inst = None
        self._fd_cache = {}
        self._idxr_cache = {}

    def get_ingest_instance(self, starting_index=0):
        """
        :return: The configuration singleton ingest instance. Type based on
            configured type field.
        :rtype: DataIngest or VideoIngest
        """
        if self._ingest_inst is None:
            self._ingest_inst = self.TYPE_MAP[self.type](self.data_dir,
                                                         self.work_dir,
                                                         starting_index)
        return self._ingest_inst

    def get_available_descriptor_labels(self):
        """
        :return: List of FeatureDescriptor configuration labels for this ingest
            configuration.
        :rtype: list of str
        """
        return self.descriptor_config['listing'].keys()

    def get_available_indexer_labels(self):
        """
        :return: List of Indexer configuration labels for this ingest
            configuration.
        :rtype: list of str
        """
        return self.indexer_config['listing'].keys()

    def get_FeatureDetector_instance(self, fd_label):
        """
        Return this configuration's FeatureDescriptor singleton instance for the
        given FeatureDescriptor type label.

        :raises KeyError: If the given label is not associated with a
            FeatureDescriptor class type.
        :raises ValueError: If the given label is not represented in the system
            configuration.

        :param fd_label: The FeatureDescriptor type label.
        :type fd_label: str

        :return: The singleton instance of the given FeatureDescriptor type for
            this configuration instance.
        :rtype: SMQTK.FeatureDescriptors.FeatureDescriptor

        """
        fd_type = get_descriptors()[fd_label]
        if fd_label not in self.get_available_descriptor_labels():
            raise ValueError("No configuration for FeatureDescriptor type '%s' "
                             "in ingest configuration '%s'"
                             % (fd_label, self.label))

        if fd_label not in self._fd_cache:
            self._fd_cache[fd_label] = fd_type(
                osp.join(self.data_dir, self.descriptor_config['dir'],
                         self.descriptor_config['listing'][fd_label]['dir']),
                osp.join(self.work_dir, self.descriptor_config['dir'],
                         self.descriptor_config['listing'][fd_label]['dir'])
            )
        return self._fd_cache[fd_label]

    def get_Indexer_instance(self, indexer_label, fd_label):
        """
        Return this configuration's Indexer singleton instance for the given
        Indexer-type/FeatureDescriptor-type label pairing.

        NOTE: This assumes a 1-to-1 relationship between descriptors and
        indexers, i.e. indexers only take features from on consistent descriptor
        source. This may change in the future depending on what kinds of
        indexers are created.

        :raises KeyError: If the given Indexer label is not associated with an
            Indexer class type.
        :raises ValueError: If the given indexer label is not represented in the
            system Indexer configuration. Also if the given FeatureDescriptor
            label is not represented in the FeatureDescriptor configuration.

        :param indexer_label: The Indexer type label.
        :type indexer_label: str

        :param fd_label: The FeatureDescriptor type label.
        :type fd_label: str

        :return: The singleton instance of the given Indexer type for this
            configuration instance.
        :rtype: SMQTK.Indexers.Indexer

        """
        idxr_type = get_indexers()[indexer_label]
        if indexer_label not in self.get_available_indexer_labels():
            raise ValueError("No configuration for Indexer type '%s' "
                             "in ingest configuration '%s'"
                             % (fd_label, self.label))
        if fd_label not in self.get_available_descriptor_labels():
            raise ValueError("No configuration for FeatureDescriptor type '%s' "
                             "in ingest configuration '%s'"
                             % (fd_label, self.label))

        if (indexer_label, fd_label) not in self._idxr_cache:
            self._idxr_cache[indexer_label, fd_label] = idxr_type(
                osp.join(self.data_dir, self.indexer_config['dir'],
                         self.indexer_config['listing'][indexer_label]['dir'],
                         self.descriptor_config['listing'][fd_label]['dir']),
                osp.join(self.work_dir, self.indexer_config['dir'],
                         self.indexer_config['listing'][indexer_label]['dir'],
                         self.descriptor_config['listing'][fd_label]['dir'])
            )
        return self._idxr_cache[indexer_label, fd_label]
