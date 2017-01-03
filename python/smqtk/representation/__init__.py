from smqtk.utils import Configurable, SmqtkObject


__all__ = [
    'SmqtkRepresentation',
    'ClassificationElement', 'get_classification_element_impls',
    'DataElement', 'get_data_element_impls',
    'DataSet', 'get_data_set_impls',
    'DescriptorElement', 'get_descriptor_element_impls',
    'DescriptorIndex', 'get_descriptor_index_impls',
    'ClassificationElementFactory', 'DescriptorElementFactory',
]


class SmqtkRepresentation (SmqtkObject, Configurable):
    """
    Interface for data representation interfaces and implementations.

    Data should be serializable, so this interface adds abstract methods for
    serializing and de-serializing SMQTK data representation instances.

    """
    # TODO(paul.tunison): Add serialization abstract method signatures here.
    # - Could start with just requiring implementing sub-classes to
    #   ``__getstate__`` and ``__setstate__`` methods required for pickle
    #   interface.


from .classification_element import ClassificationElement, \
    get_classification_element_impls
from .data_element import DataElement, get_data_element_impls
from .data_set import DataSet, get_data_set_impls
from .descriptor_element import DescriptorElement, get_descriptor_element_impls
from .descriptor_index import DescriptorIndex, get_descriptor_index_impls
from .key_value import KeyValueStore, get_key_value_store_impls

from .classification_element_factory import ClassificationElementFactory
from .descriptor_element_factory import DescriptorElementFactory
