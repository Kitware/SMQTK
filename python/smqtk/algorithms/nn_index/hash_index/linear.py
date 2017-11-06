import heapq

from six.moves import StringIO
import numpy

from smqtk.algorithms.nn_index.hash_index import HashIndex
from smqtk.representation import get_data_element_impls
from smqtk.utils import merge_dict, plugin
from smqtk.utils.bit_utils import (
    bit_vector_to_int_large,
    int_to_bit_vector_large,
)
from smqtk.utils.metrics import hamming_distance


class LinearHashIndex (HashIndex):
    """
    Basic linear index using heap sort (aka brute force).
    Hash codes are stored as large integer values.
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as arguments,
        turning those argument names into configuration dictionary keys. If any
        of those arguments have defaults, we will add those values into the
        configuration dictionary appropriately. The dictionary returned should
        only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        c = super(LinearHashIndex, cls).get_default_config()
        c['cache_element'] = plugin.make_config(get_data_element_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless an instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: LinearHashIndex

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        cache_element = None
        if config_dict['cache_element'] \
                and config_dict['cache_element']['type']:
            cache_element = \
                plugin.from_plugin_config(config_dict['cache_element'],
                                          get_data_element_impls())
        config_dict['cache_element'] = cache_element

        return super(LinearHashIndex, cls).from_config(config_dict, False)

    def __init__(self, cache_element=None):
        """
        Initialize linear, brute-force hash index.

        :param cache_element: Optional data element to cache our index to.
        :type cache_element: smqtk.representation.DataElement | None

        """
        super(LinearHashIndex, self).__init__()
        self.cache_element = cache_element
        self.index = set()
        self.load_cache()

    def get_config(self):
        c = self.get_default_config()
        if self.cache_element:
            c['cache_element'] = merge_dict(c['cache_element'],
                                            plugin.to_plugin_config(
                                                self.cache_element))
        return c

    def load_cache(self):
        """
        Load from file cache if we have one
        """
        if self.cache_element and not self.cache_element.is_empty():
            buff = StringIO(self.cache_element.get_bytes())
            self.index = set(numpy.load(buff))

    def save_cache(self):
        """
        save to file cache if configures
        """
        if self.cache_element and self.index:
            if self.cache_element.is_read_only():
                raise ValueError("Cache element (%s) is read-only."
                                 % self.cache_element)
            buff = StringIO()
            # noinspection PyTypeChecker
            numpy.save(buff, tuple(self.index))
            self.cache_element.set_bytes(buff.getvalue())

    def count(self):
        return len(self.index)

    def _build_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to build the index with
        the given hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :param hashes: Iterable of descriptor elements to build index
            over.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """
        new_index = set(map(bit_vector_to_int_large, hashes))
        self.index = new_index
        self.save_cache()

    def _update_index(self, hashes):
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more hash vectors given.

        If no index exists yet, a new one should be created using the given hash
        vectors.

        :param hashes: Iterable of numpy boolean hash vectors to add to this
            index.
        :type hashes: collections.Iterable[numpy.ndarray[bool]]

        """
        self.index.update(set(map(bit_vector_to_int_large, hashes)))
        self.save_cache()

    def _nn(self, h, n=1):
        """
        Internal method to be implemented by sub-classes to return the nearest
        `N` neighbor hash codes as bit-vectors to the given hash code
        bit-vector.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query (normalized hamming distance).

        When this internal method is called, we have already checked that our
        index is not empty.

        :param h: Hash code to compute the neighbors of. Should be the same bit
            length as indexed hash codes.
        :type h: numpy.ndarray[bool]

        :param n: Number of nearest neighbors to find.
        :type n: int

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.
        :rtype: (tuple[numpy.ndarray[bool]], tuple[float])

        """
        h_int = bit_vector_to_int_large(h)
        bits = len(h)
        #: :type: list[int|long]
        near_codes = \
            heapq.nsmallest(n, self.index,
                            lambda e: hamming_distance(h_int, e)
                            )
        distances = map(hamming_distance, near_codes,
                        [h_int] * len(near_codes))
        return [int_to_bit_vector_large(c, bits) for c in near_codes], \
               [d / float(bits) for d in distances]
