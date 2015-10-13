from smqtk.utils import Configurable, SmqtkObject


class SmqtkRepresentation (SmqtkObject, Configurable):
    """
    Interface for data representation interfaces and implementations.

    Data should be serializable, so this interface adds abstract methods for
    serializing and de-serializing SMQTK data representation instances.

    """
    # TODO: Add serialization abstract method signatures here


from .code_index import CodeIndex, get_code_index_impls
from .data_element import DataElement, get_data_element_impls
from .data_set import DataSet, get_data_set_impls
from .descriptor_element import DescriptorElement, get_descriptor_element_impls
from .descriptor_index import DescriptorIndex, get_descriptor_index_impls

from .descriptor_element_factory import DescriptorElementFactory
