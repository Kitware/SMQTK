import threading
import time
import six

from smqtk.representation import DescriptorElement
from smqtk.representation import DescriptorElementFactory
from smqtk.representation import get_descriptor_element_impls


__author__ = 'paul.tunison@kitware.com'


class CachingDescriptorElement (DescriptorElement):

    @classmethod
    def is_usable(cls):
        """
        This implementation has no direct dependencies of its own.
        :rtype: bool
        """
        return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as arguments,
        aside from the first two assumed positional arguments, turning those
        argument names into configuration dictionary keys.
        If any of those arguments have defaults, we will add those values into
        the configuration dictionary appropriately.
        The dictionary returned should only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        c = super(CachingDescriptorElement, cls).get_default_config()

        # Nested DescriptorElementFactory configuration
        if c['wrapped_element_factory'] is None:
            # Have to make this configuration in such a way that we don't
            # include ourselves in the list of nestable classes else an infinite
            # recursion will occur.

            de_impls = get_descriptor_element_impls()
            # Remove ourselves
            del de_impls[cls.__name__]

            # Construct config block DescriptorElementFactory wants
            c_def = {"type": None}
            for label, de_cls in six.iteritems(de_impls):
                # noinspection PyUnresolvedReferences
                c_def[label] = de_cls.get_default_config()
            c['wrapped_element_factory'] = c_def
        else:
            c['wrapped_element_factory'] = \
                c['wrapped_element_factory'].get_config()

        return c

    @classmethod
    def from_config(cls, config_dict, type_str, uuid, merge_default=True):
        # convert factory configuration
        config_dict['wrapped_element_factory'] = \
            DescriptorElementFactory.from_config(
                config_dict['wrapped_element_factory']
            )

        return super(CachingDescriptorElement, cls).from_config(
            config_dict, type_str, uuid, merge_default
        )

    def __init__(self, type_str, uuid, wrapped_element_factory,
                 cache_expiration_timeout=1.0, poll_interval=0.1):
        """
        Initialize a new caching wrapper descriptor element.

        This implementation is intended to wrap another DescriptorElement type,
        adding a timed caching layer on top of

        :raises AssertionError: Cache expiration seconds was not a positive
            value.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :type type_str: str

        :param uuid: Unique ID reference of the descriptor.
        :type uuid: collections.Hashable

        :param wrapped_element_factory: DescriptorElementFactory to produce
            DescriptorElement instances of the wrapped type.
        :type wrapped_element_factory:
            smqtk.representation.DescriptorElementFactory

        :param cache_expiration_timeout: Timeout in seconds for accessed
            descriptors to be cached. If this is non-zero, a monitoring thread
            will be launched in order to track the timeout. The thread will be
            shutdown after timeout. This value must be positive.

            If this is positive infinity, then the cache never expires. This
            also means that the cache will not be updated if the vector that
            would be returned from the wrapped element ever changes.
        :type cache_expiration_timeout: None | float

        :param poll_interval: How often to check if the cache has expired.
        :type poll_interval: float

        """
        super(CachingDescriptorElement, self).__init__(type_str, uuid)

        self.wrapped_element_factory = wrapped_element_factory
        self.cache_expiration_timeout = float(cache_expiration_timeout)
        self.poll_interval = poll_interval

        assert cache_expiration_timeout > 0, \
            "Cache expiration timeout was not positive."

        self._d_elem = self.wrapped_element_factory \
                           .new_descriptor(self.type(), self.uuid())
        # self._log.debug("Caching descriptor element instance of type '%s'",
        #                 self._d_elem.__class__.__name__)

        # Attributes for timed caching with threads
        self.cache_v = None  # Numpy ndarray if there is a current cache
        self.cache_last_access = None  # UNIX timestamp
        self.cache_lock = threading.RLock()
        #: :type: threading.Thread
        self.cache_thread = None  # the expiry monitor if there is a cached v

    def __del__(self):
        """
        Release vector cache
        """
        with self.cache_lock:
            # Should cause dependent thread to terminate gracefully
            self.cache_v = None
        if self.cache_thread and \
                self.cache_thread.ident != threading.currentThread():
            self.cache_thread.join()

    def __getstate__(self):
        return {
            # Base DescriptorElement stuff
            "type": self.type(),
            "uuid": self.uuid(),
            # This impl's stuff
            "wrapped_element_factory": self.wrapped_element_factory,
            "cache_expiration_timeout": self.cache_expiration_timeout,
            "poll_interval": self.poll_interval
        }

    def __setstate__(self, c):
        # base-class
        self._type_label = c['type']
        self._uuid = c['uuid']
        # this-class
        self.wrapped_element_factory = c['wrapped_element_factory']
        self.cache_expiration_timeout = c['cache_expiration_timeout']
        self.poll_interval = c['poll_interval']

        # Initializing local cache variables that were un-pickle-able
        self._d_elem = self.wrapped_element_factory\
                           .new_descriptor(self.type(), self.uuid())
        self.cache_v = None
        self.cache_last_access = None
        self.cache_lock = threading.RLock()
        self.cache_thread = None

    def get_config(self):
        return {
            "wrapped_element_factory": self.wrapped_element_factory,
            "cache_expiration_timeout": self.cache_expiration_timeout,
            "poll_interval": self.poll_interval,
        }

    def has_vector(self):
        """
        :return: Whether or not this container current has a descriptor vector
            stored.
        :rtype: bool
        """
        return self.vector() is not None

    def vector(self):
        """
        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None
        """
        with self.cache_lock:
            v = self.cache_v

            # If no cache, attempt to populate it
            if v is None:

                # No cache currently, attempt fetch from wrapped elem
                # self._log.debug("Getting vector from nested descriptor "
                #                 "element")
                v = self._d_elem.vector()
                # self._log.debug("Vector received: %s", v)

                if v is not None:
                    # Clean-up old thread if there was one
                    if self.cache_thread:
                        # cache_v is None at this point
                        # self._log.debug("Joining old monitor thread")
                        self.cache_lock.release()
                        self.cache_thread.join()
                        self.cache_lock.acquire()

                    # vector in elem; set in cache; start monitor thread
                    self.cache_v = v
                    self.cache_thread = threading.Thread(
                        target=CachingDescriptorElement
                        .thread_monitor_cache_expiration,
                        args=(self,),
                        # verbose=self._log.getEffectiveLevel()<=logging.DEBUG,
                    )
                    # self._log.debug("Spawning cache monitor thread")
                    self.cache_thread.start()

            else:
                # Currently have a cache
                assert self.cache_thread is not None, \
                    "Have a cache, but no monitor thread."

            self.cache_last_access = time.time()

        return v

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray

        """
        # set source vector and set as current cache
        with self.cache_lock:
            # self._log.debug("Setting in source element")
            self._d_elem.set_vector(new_vec)

            # Only start monitor when we're given a cache-able vector and the
            #   cache is currently not occupied.
            if new_vec is not None and self.cache_v is None:
                if self.cache_thread:
                    # cache_thread not alive at this point
                    # self._log.debug("Joining old monitor thread")
                    self.cache_lock.release()
                    self.cache_thread.join()
                    self.cache_lock.acquire()

                # need to start new monitor thread
                self.cache_thread = threading.Thread(
                    target=CachingDescriptorElement
                    .thread_monitor_cache_expiration,
                    args=(self,),
                    # verbose=self._log.getEffectiveLevel() <= logging.DEBUG,
                )
                # self._log.debug("Spawning cache monitor thread")
                self.cache_thread.start()

            # Update cache vector and access time
            self.cache_v = new_vec
            self.cache_last_access = time.time()

    @staticmethod
    def thread_monitor_cache_expiration(elem):
        """
        Monitor wrapper element's cache expiration

        :param elem: Wrapper element instance for monitoring
        :type elem: CachingDescriptorElement

        """
        # log = logging.getLogger(__name__)
        #
        # # noinspection PyProtectedMember
        # log_header = '[{type:s}, {uuid:s}, {elem:s}]'.format(**{
        #     "type": elem.type(),
        #     'uuid': elem.uuid(),
        #     'elem': elem.wrapped_element_factory._d_type,
        # })

        expired = False
        while not expired:
            time.sleep(elem.poll_interval)
            with elem.cache_lock:
                t = time.time()
                # log.debug("%s Checking cache cache expiration "
                #           "[now = %f | last access = %f | timeout = %f]",
                #           log_header, t, elem.cache_last_access,
                #           elem.cache_expiration_timeout)
                if t - elem.cache_last_access >= elem.cache_expiration_timeout:
                    elem.cache_v = None
                    expired = True
                    # log.debug("%s Cache expired", log_header)
                elif elem.cache_v is None:
                    expired = True
                    # log.debug("%s Cache was invalidated for us", log_header)
        # log.debug("%s Monitor thread exiting", log_header)


# Disabling this implementation for the moment because it needs to be rethought
DESCRIPTOR_ELEMENT_CLASS = None
