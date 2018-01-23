"""
Interface and plugin getter for LSH algorithm hash generation functors.
"""
import abc
import os

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.utils import plugin


class LshFunctor (SmqtkAlgorithm):
    """
    Locality-sensitive hashing functor interface.

    The aim of such a function is to be able to generate hash codes
    (bit-vectors) such that similar items map to the same or similar hashes
    with a high probability. In other words, it aims to maximize hash collision
    for similar items.

    **Building Models**

    Some hash functions want to build a model based on some training set of
    descriptors. Due to the non-standard nature of algorithm training and model
    building, please refer to the specific implementation for further
    information on whether model training is needed and how it is accomplished.

    """

    def __call__(self, descriptor):
        return self.get_hash(descriptor)

    @abc.abstractmethod
    def get_hash(self, descriptor):
        """
        Get the locality-sensitive hash code for the input descriptor.

        :param descriptor: Descriptor vector we should generate the hash of.
        :type descriptor: numpy.ndarray[float]

        :return: Generated bit-vector as a numpy array of booleans.
        :rtype: numpy.ndarray[bool]

        """


def get_lsh_functor_impls(reload_modules=False):
    """
    Discover and return discovered ``LshFunctor`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable
          :envvar:`LSH_FUNCTOR_PATH`
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``LSH_FUNCTOR_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type :class:`.LshFunctor`
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "LSH_FUNCTOR_PATH"
    helper_var = "LSH_FUNCTOR_CLASS"
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var,
                              LshFunctor, reload_modules=reload_modules)
