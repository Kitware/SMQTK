import abc
import inspect
import types

from smqtk.utils import merge_dict


class Configurable (object):
    """
    Interface for objects that should be configurable via a configuration
    dictionary consisting of JSON types.
    """
    __metaclass__ = abc.ABCMeta

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
        if isinstance(cls.__init__, types.MethodType):
            argspec = inspect.getargspec(cls.__init__)

            # Ignores potential *args or **kwargs present
            params = argspec.args[1:]  # skipping ``self`` arg
            num_params = len(params)

            if argspec.defaults:
                num_defaults = len(argspec.defaults)
                vals = ((None,) * (num_params - num_defaults) + argspec.defaults)
            else:
                vals = (None,) * num_params

            return dict(zip(params, vals))

        # No constructor explicitly defined on this class
        return {}

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
        :rtype: Configurable

        """
        # The simple case is that the class doesn't require any special
        # parameters other than those that can be provided via the JSON
        # specification, which we cover here. If an implementation needs
        # something more special, they can override this function.
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        return cls(**config_dict)

    @abc.abstractmethod
    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
