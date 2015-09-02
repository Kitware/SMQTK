"""
Helper classes for access to JSON system configuration
"""

import abc
import smqtk_config

from smqtk.data_rep.data_set_impl import get_data_set_impls
from smqtk.data_rep.descriptor_element_impl import get_descriptor_element_impls
from smqtk.data_rep.descriptor_element_factory import DescriptorElementFactory
from smqtk.content_description import get_descriptors
from smqtk.indexing import get_indexers
from smqtk.similarity_index.lsh.code_index import get_index_types


def str_to_bool(s):
    return s.lower() in {'yes', 'true'}


class ConfigurationInterface (object):
    """
    Abstract interface for constructing object instances based central on system
    configuration JSON file.
    """
    __metaclass__ = abc.ABCMeta

    # This is the base JSON configuration dictionary. When using a custom
    # configuration dictionary, it should be set here, and will be reflected in
    # the use of the other configuration classes.
    BASE_CONFIG = smqtk_config.SYSTEM_CONFIG

    @classmethod
    @abc.abstractmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return

    @classmethod
    @abc.abstractmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return

    @classmethod
    @abc.abstractmethod
    def new_inst(cls, label):
        """
        Construct a new instance of the type and with parameters associated with
        the given label.

        :param label: the configuration label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New instance of type and parameters associated with the given
            label.

        """
        return


class DataSetConfiguration (ConfigurationInterface):
    """
    Interface into data set configurations in common system configuration file
    """

    CFG_SECT = 'DataSets'

    @classmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return cls.BASE_CONFIG[cls.CFG_SECT]

    @classmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return set(cls.get_config_sect())

    @classmethod
    def new_inst(cls, label):
        """
        :param label: the data set label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New data set instance.
        :rtype: smqtk.data_rep.DataSet

        """
        label_sect = cls.get_config_sect()[label]
        ds_cls = get_data_set_impls()[label_sect['type']]
        return ds_cls(**label_sect['init'])


class DescriptorFactoryConfiguration (ConfigurationInterface):
    """
    Interface into DescriptorElementFactory configurations in the common system
    configuration file.
    """

    CFG_SECT = "DescriptorElementFactories"

    @classmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return cls.BASE_CONFIG[cls.CFG_SECT]

    @classmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return cls.get_config_sect().keys()

    @classmethod
    def new_inst(cls, label):
        """
        Construct a new instance of the type and with parameters associated with
        the given label.

        :param label: the configuration label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New instance of type and parameters associated with the given
            label.

        """
        label_sect = cls.get_config_sect()[label]
        d_type = get_descriptor_element_impls()[label_sect['type']]
        init_params = label_sect['init']
        return DescriptorElementFactory(d_type, init_params)


class ContentDescriptorConfiguration (ConfigurationInterface):
    """
    Interface into ContentDescriptor configurations in the common system
    configuration file.
    """

    CFG_SECT = 'ContentDescriptors'

    @classmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return cls.BASE_CONFIG[cls.CFG_SECT]

    @classmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return set(cls.get_config_sect())

    @classmethod
    def new_inst(cls, label):
        """
        Construct a new instance of the type and with parameters associated with
        the given label.

        :param label: the configuration label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New instance of type and parameters associated with the given
            label.
        :rtype: smqtk.content_description.ContentDescriptor

        """
        label_sect = cls.get_config_sect()[label]
        cd_cls = get_descriptors()[label_sect['type']]
        return cd_cls(**label_sect['init'])


class LSHCodeIndexConfiguration (ConfigurationInterface):

    CFG_SECT = 'LSHCodeIndices'

    @classmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return cls.BASE_CONFIG[cls.CFG_SECT]

    @classmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return set(cls.get_config_sect())

    @classmethod
    def new_inst(cls, label):
        """
        Construct a new instance of the type and with parameters associated with
        the given label.

        :param label: the configuration label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New instance of type and parameters associated with the given
            label.
        :rtype: smqtk.content_description.ContentDescriptor

        """
        label_sect = cls.get_config_sect()[label]
        cd_cls = get_index_types()[label_sect['type']]
        return cd_cls(**label_sect['init'])


###
# Deprecating soon
#

class IndexerConfiguration (ConfigurationInterface):
    """
    Interface into Indexer configurations in the common system configuration
    file.
    """

    CFG_SECT = "Indexers"

    @classmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return cls.BASE_CONFIG[cls.CFG_SECT]

    @classmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return set(cls.get_config_sect())

    @classmethod
    def new_inst(cls, label):
        """
        Construct a new instance of the type and with parameters associated with
        the given label.

        :param label: the configuration label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New instance of type and parameters associated with the given
            label.
        :rtype: smqtk.indexing.Indexer

        """
        label_sect = cls.get_config_sect()[label]
        idxr_class = get_indexers()[label_sect['type']]
        return idxr_class(**label_sect['init'])


class FusionConfiguration (ConfigurationInterface):
    """
    Interface into Fusion system configuration in the common system
    configuration file.
    """

    CFG_SECT = "Fusion"

    @classmethod
    def get_config_sect(cls):
        """
        :return: Dictionary configuration block for this configuration
            component.
        :rtype: dict
        """
        return cls.BASE_CONFIG[cls.CFG_SECT]

    @classmethod
    def available_labels(cls):
        """
        :return: Set of available string labels in system configuration.
        :rtype: set[str]
        """
        return set(cls.get_config_sect())

    @classmethod
    def new_inst(cls, label):
        """
        Construct a new instance of the type and with parameters associated with
        the given label.

        :param label: the configuration label
        :type label: str

        :raises KeyError: The given label does not exist in the system
            configuration

        :return: New instance of type and parameters associated with the given
            label.
        :rtype: smqtk.fusion.reactor.Reactor

        """
        from smqtk.fusion.atom import Atom
        from smqtk.fusion.reactor import Reactor
        from smqtk.fusion.catalyst import get_catalysts

        label_sect = cls.get_config_sect()[label]
        catalyst_inst = get_catalysts()[label_sect['catalyst']['type']](
            **label_sect['catalyst']['init']
        )

        atom_inst_list = []
        for atom_sect in label_sect['atoms']:
            atom_inst_list.append(Atom(
                ContentDescriptorConfiguration.new_inst(atom_sect['descriptor']),
                [IndexerConfiguration.new_inst(ilabel)
                 for ilabel in atom_sect['indexers']]
            ))

        reactor_inst = Reactor(atom_inst_list, catalyst_inst)
        return reactor_inst
