from smqtk.utils import SmqtkObject
from smqtk.utils import Configurable, plugin


__all__ = [
    'SmqtkAlgorithm',
    'Classifier', 'SupervisedClassifier', 'get_classifier_impls',
    'DescriptorGenerator', 'get_descriptor_generator_impls',
    'NearestNeighborsIndex', 'get_nn_index_impls',
    'HashIndex', 'get_hash_index_impls',
    'LshFunctor', 'get_lsh_functor_impls',
    'RelevancyIndex', 'get_relevancy_index_impls',
]


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
from .classifier import Classifier, SupervisedClassifier, get_classifier_impls
from .descriptor_generator import DescriptorGenerator, get_descriptor_generator_impls
from .nn_index import NearestNeighborsIndex, get_nn_index_impls
from .nn_index.hash_index import HashIndex, get_hash_index_impls
from .nn_index.lsh.functors import LshFunctor, get_lsh_functor_impls
from .relevancy_index import RelevancyIndex, get_relevancy_index_impls
