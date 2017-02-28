import abc
import os

from smqtk.algorithms import NearestNeighborsIndex
from smqtk.utils.plugin import get_plugins


class HashIndex (NearestNeighborsIndex):
    """
    Specialized ``NearestNeighborsIndex`` for indexing unique hash codes
    bit-vectors) in memory (numpy arrays) using the hamming distance metric.

    Implementations of this interface cannot be used in place of something
    requiring a ``NearestNeighborsIndex`` implementation due to the speciality
    of this interface.

    Only unique bit vectors should be indexed. The ``nn`` method should not
    return the same bit vector more than once for any query.
    """

    @abc.abstractmethod
    def build_index(self, hashes):
        """
        Build the index with the give hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the index, not add to
        it. If an exception is raised, the current index, if there is one, will
        not be modified.

        :raises ValueError: No data available in the given iterable.

        :param hashes: Iterable of descriptor elements to build index
            over.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """

    @abc.abstractmethod
    def nn(self, h, n=1):
        """
        Return the nearest `N` neighbor hash codes as bit-vectors to the given
        hash code bit-vector.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query (normalized hamming distance).

        :param h: Hash code to compute the neighbors of. Should be the same bit
            length as indexed hash codes.
        :type h: numpy.ndarray[bool]

        :param n: Number of nearest neighbors to find.
        :type n: int

        :raises ValueError: No index to query from.

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.
        :rtype: (tuple[numpy.ndarray[bool]], tuple[float])

        """
        if not self.count():
            raise ValueError("No index currently set to query from!")


def get_hash_index_impls(reload_modules=False):
    """
    Discover and return discovered ``HashIndex`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable ``HASH_INDEX_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``HASH_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``HashIndex``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "HASH_INDEX_PATH"
    helper_var = "HASH_INDEX_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var,
                       HashIndex, reload_modules=reload_modules)
