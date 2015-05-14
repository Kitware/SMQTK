"""
Helper classes for access to JSON system configuration
"""


import os.path as osp
import smqtk_config

from SMQTK.content_description import get_descriptors
from SMQTK.Fusion.Atom import Atom
from SMQTK.Fusion.Catalyst import get_catalysts
from SMQTK.Fusion.Reactor import Reactor
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

    #: :type: dict
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
    def base_ingest_data_dir(cls):
        """
        :return: The base directory where ingests data should be located
        :rtype: str
        """
        return osp.join(smqtk_config.DATA_DIR, cls.INGEST_CONFIG['dir'])

    @classmethod
    def base_ingest_work_dir(cls):
        """
        :return: The base directory where ingests work should be located
        :rtype: str
        """
        return osp.join(smqtk_config.WORK_DIR, cls.INGEST_CONFIG['dir'])

    def __init__(self, ingest_label, config_dict=None):
        """
        :param ingest_label: Ingest to configure to
        :type ingest_label: str

        :param config_dict: Custom configuration dictionary to use instead of
            common system configuration JSON file contents.
        :type config_dict: dict

        """
        # Override local base config dict if one was given
        self.INGEST_CONFIG = config_dict or IngestConfiguration.INGEST_CONFIG

        if ingest_label not in self.available_ingest_labels():
            raise ValueError("Given ingest label '%s' not available in "
                             "configuration! Make sure to add configuration "
                             "for an ingest first."
                             % ingest_label)

        label_config = self.INGEST_CONFIG['listing'][ingest_label]

        self.label = ingest_label
        self.data_dir = osp.join(self.base_ingest_data_dir(), label_config['dir'])
        self.work_dir = osp.join(self.base_ingest_work_dir(), label_config['dir'])
        self.type = label_config['type']

        self.descriptor_config = label_config['descriptors']
        self.indexer_config = label_config['indexers']
        self.fusion_config = label_config['fusion']

    def new_ingest_instance(self, data_dir=None, work_dir=None,
                            starting_index=0):
        """


        :param data_dir: Data directory override.
        :type data_dir: str

        :param work_dir: Working directory override
        :type work_dir: str

        :return: The configuration singleton ingest instance. Type based on
            configured type field.
        :rtype: DataIngest or VideoIngest
        """
        return self.TYPE_MAP[self.type](data_dir or self.data_dir,
                                        work_dir or self.work_dir,
                                        starting_index)

    def get_available_descriptor_labels(self):
        """
        :return: List of FeatureDescriptor configuration labels for this ingest
            configuration.
        :rtype: list of str
        """
        return self.descriptor_config['listing'].keys()

    def new_descriptor_instance(self, fd_label, data_dir=None, work_dir=None):
        """
        Get a new descriptor instance.

        :raises KeyError: If the given label is not associated with a
            FeatureDescriptor class type.
        :raises ValueError: If the given label is not represented in the system
            configuration.

        :param fd_label: The FeatureDescriptor type label.
        :type fd_label: str

        :param data_dir: Data directory override, otherwise uses configured
            directory.
        :type data_dir: str

        :param work_dir: Working directory override, otherwise uses configured
            directory.
        :type work_dir: str

        :return: New instance of the given FeatureDescriptor type for this
            configuration instance.
        :rtype: SMQTK.FeatureDescriptors.FeatureDescriptor

        """
        fd_type = get_descriptors()[fd_label]
        if fd_label not in self.get_available_descriptor_labels():
            raise ValueError("No configuration for FeatureDescriptor type '%s' "
                             "in ingest configuration '%s'"
                             % (fd_label, self.label))

        return fd_type(
            data_dir or
            osp.join(self.data_dir, self.descriptor_config['dir'],
                     self.descriptor_config['listing'][fd_label]['dir'])
            ,
            work_dir or
            osp.join(self.work_dir, self.descriptor_config['dir'],
                     self.descriptor_config['listing'][fd_label]['dir'])
        )

    def get_available_indexer_labels(self):
        """
        :return: List of Indexer configuration labels for this ingest
            configuration.
        :rtype: list of str
        """
        return self.indexer_config['listing'].keys()

    def new_indexer_instance(self, indexer_label, fd_label,
                             data_dir=None, work_dir=None):
        """
        Get a new indexer instance.

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

        :param data_dir: Data directory override, otherwise uses configured
            directory.
        :type data_dir: str

        :param work_dir: Working directory override, otherwise uses configured
            directory.
        :type work_dir: str

        :return: New instance of the given Indexer type for this configuration
            instance.
        :rtype: SMQTK.Indexers.Indexer

        """
        idxr_type = get_indexers()[indexer_label]
        if indexer_label not in self.get_available_indexer_labels():
            raise ValueError("No configuration for Indexer type '%s' "
                             "in ingest configuration '%s'"
                             % (indexer_label, self.label))
        if fd_label not in self.get_available_descriptor_labels():
            raise ValueError("No configuration for FeatureDescriptor type '%s' "
                             "in ingest configuration '%s'"
                             % (fd_label, self.label))

        return idxr_type(
            data_dir or
            osp.join(self.data_dir, self.indexer_config['dir'],
                     self.indexer_config['listing'][indexer_label]['dir'],
                     self.descriptor_config['listing'][fd_label]['dir'])
            ,
            work_dir or
            osp.join(self.work_dir, self.indexer_config['dir'],
                     self.indexer_config['listing'][indexer_label]['dir'],
                     self.descriptor_config['listing'][fd_label]['dir'])
        )

    def get_available_catalyst_labels(self):
        """
        :return: List of Catalyst configuration labels for this ingest
            configuration.
        :rtype: list of str
        """
        return self.fusion_config['catalysts']['listing'].keys()

    def new_catalyst_instance(self, catalyst_label, data_dir=None,
                              work_dir=None):
        """
        Get a new Catalyst instance.

        :param catalyst_label: The Catalyst type label.
        :type catalyst_label: str

        :param data_dir: Data directory override, otherwise uses configured
            directory.
        :type data_dir: str

        :param work_dir: Working directory override, otherwise uses configured
            directory.
        :type work_dir: str

        :return: New fusion Catalyst instance.
        :rtype: SMQTK.Fusion.Catalyst.Catalyst

        """
        catalyst_type = get_catalysts()[catalyst_label]
        catalyst_config = self.fusion_config['catalysts']
        if catalyst_label not in catalyst_config['listing']:
            raise ValueError("No configuration for Catalyst type '%s' in "
                             "ingest configuration '%s'"
                             % (catalyst_label, self.label))

        return catalyst_type(
            data_dir or
            osp.join(self.data_dir, catalyst_config['dir'],
                     catalyst_config['listing'][catalyst_label]['dir'])
            ,
            work_dir or
            osp.join(self.work_dir, catalyst_config['dir'],
                     catalyst_config['listing'][catalyst_label]['dir'])
        )

    def new_reactor(self, catalyst_label):
        """
        Construct a new descriptor/indexer Reactor instance as configured for
        this ingest. This creates new instances of descriptors and indexers

        :param catalyst_label: The Catalyst type label.
        :type catalyst_label: str

        :return: New SMQTK Reactor instance
        :rtype: SMQTK.Fusion.Reactor.Reactor

        """
        # Build from "fusion" section under ingest configuration
        fusion_catalyst = self.new_catalyst_instance(catalyst_label)

        #: :type: list of dict
        atom_configs = self.fusion_config['atoms']
        atom_instances = []
        for a_config in atom_configs:
            d_label = a_config['descriptor']
            descriptor = self.new_descriptor_instance(d_label)
            indexers = []
            for i_label in a_config['indexers']:
                indexers.append(self.new_indexer_instance(i_label, d_label))
            atom_instances.append(Atom(descriptor, indexers))

            # TODO: Collect sub-catalyst instances if/when specified

        return Reactor(atom_instances, fusion_catalyst)
