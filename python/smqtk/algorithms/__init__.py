import logging

from smqtk.utils import SmqtkObject
from smqtk.utils import Configurable, plugin


__author__ = "paul.tunison@kitware.com"


class SmqtkAlgorithm (SmqtkObject, Configurable, plugin.Pluggable):
    """
    Parent class for all algorithm interfaces.
    """

    @property
    def name(self):
        """
        :return: The name of this class type.
        :rtype: str
        """
        return self.__class__.__name__


# Import module abstracts and plugin getter functions
from .descriptor_generator import DescriptorGenerator, get_descriptor_generator_impls
from .nn_index import NearestNeighborsIndex, get_nn_index_impls
from .relevancy_index import RelevancyIndex, get_relevancy_index_impls
