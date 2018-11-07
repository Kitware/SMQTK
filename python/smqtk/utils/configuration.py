"""
Helper interface and functions for generalized object configuration, to and from
JSON-compliant dictionaries.

While this interface and utility methods should be general enough to add
JSON-compliant dictionary-based configuration to any object, this was created
in mind with the SMQTK plugin module.

Standard configuration dictionaries take the following general format:

.. code-block:: json

    {
        "type": "one-of-the-keys-below",
        "ClassName1": {
            "param1": "val1",
            "param2": "val2"
        },
        "ClassName2": {
            "p1": 4.5,
            "p2": null
        }
    }

The "type" key is considered a special key that should always be present and it
specifies one of the other keys within the same dictionary. Each other key in
the dictionary should be the name of a ``Configurable`` inheriting class type.

"""
import abc
import inspect
import types

import six

from smqtk.utils.dict import merge_dict


@six.add_metaclass(abc.ABCMeta)
class Configurable (object):
    """
    Interface for objects that should be configurable via a configuration
    dictionary consisting of JSON types.
    """

    __slots__ = ()

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

        >>> # noinspection PyUnresolvedReferences
        >>> class SimpleConfig(Configurable):
        ...     def __init__(self, a=1, b='foo'):
        ...         self.a = a
        ...         self.b = b
        ...     def get_config(self):
        ...         return {'a': self.a, 'b': self.b}
        >>> self = SimpleConfig()
        >>> config = self.get_default_config()
        >>> assert config == {'a': 1, 'b': 'foo'}
        """
        if isinstance(cls.__init__, (types.MethodType, types.FunctionType)):
            argspec = inspect.getargspec(cls.__init__)

            # Ignores potential *args or **kwargs present
            params = argspec.args[1:]  # skipping ``self`` arg
            num_params = len(params)

            if argspec.defaults:
                num_defaults = len(argspec.defaults)
                vals = ((None,) * (num_params - num_defaults)
                        + argspec.defaults)
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

        # noinspection PyArgumentList
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


def make_default_config(configurable_iter):
    """
    Generated default configuration dictionary for the given iterable of
    Configurable-inheriting types.

    for the given map of configurable class
    types (as would normally be returned by ``smqtk.utils.plugin.get_plugins``).

    A types parameters, as listed, at the construction parameters for that
    type. Default values are inserted where possible, otherwise None values are
    used.

    :param collections.Iterable[type] configurable_iter:
        A dictionary mapping class names to class types.

    :return: Base configuration dictionary with an empty ``type`` field, and
        containing the types and initialization parameter specification for all
        implementation types available from the provided getter method.
    :rtype: dict[str, object]

    """
    d = {"type": None}
    for cls in configurable_iter:
        assert isinstance(cls, type) and issubclass(cls, Configurable), \
            "Encountered invalid Configurable type: '{}' (type={})".format(
                cls, type(cls)
            )
        d[cls.__name__] = cls.get_default_config()
    return d


def to_config_dict(c_inst):
    """
    Helper function that transforms the configuration dictionary retrieved from
    ``configurable_inst`` into the "standard" SMQTK configuration dictionary
    format (see above).

    :param Configurable c_inst:
        Instance of a class type that subclasses the ``Configurable`` interface.

    :return: Standard format configuration dictionary.
    :rtype: dict

    """
    c_class = c_inst.__class__
    assert not isinstance(c_inst, type) and \
        issubclass(c_class, Configurable), \
        "c_inst must be an instance and its type must subclass from " \
        "Configurable."
    c_class_name = c_class.__name__
    return {
        "type": c_class_name,
        c_class_name: c_inst.get_config()
    }


def from_config_dict(config, type_iter, *args):
    """
    Helper function for instantiating an instance of a class given the
    configuration dictionary ``config`` from available types provided by
    ``type_iter`` via the ``Configurable`` interface's ``from_config``
    class-method.

    ``args`` are additionally positional arguments to be passed to the type's
    ``from_config`` method on return.

    :raises ValueError:
        This may be raised if:
            - type field set to ``None``
            - type field did not match any available configuration in the given
              config.
            - Type field did not specify any implementation key.

    :raises TypeError: Insufficient/incorrect initialization parameters were
        specified for the specified ``type``'s constructor.

    :param dict config:
        Configuration dictionary to draw from.

    :param collections.Iterable[type] type_iter:
        An iterable of class types to select from.

    :return: Instance of the configured class type as specified in ``config``
        and as available in ``type_iter``.
    :rtype: smqtk.utils.configuration.Configurable

    """
    if 'type' not in config:
        raise ValueError("Configuration dictionary given does not have an "
                         "implementation type specification.")
    conf_type = config['type']
    type_map = dict(map(lambda t: (t.__name__, t), type_iter))

    conf_type_options = set(config.keys()) - {'type'}
    iter_type_options = set(type_map.keys())
    # Type provided may either by None, not have a matching block in the config,
    # not have a matching implementation type, or match both.
    if conf_type is None:
        raise ValueError("No implementation type specified. Options: %s"
                         % conf_type_options)
    elif conf_type not in conf_type_options:
        raise ValueError("Implementation type specified as '%s', but no "
                         "configuration block was present for that type. "
                         "Available configuration block options: %s"
                         % (conf_type, list(conf_type_options)))
    elif conf_type not in iter_type_options:
        raise ValueError("Implementation type specified as '%s', but no "
                         "plugin implementations are available for that type. "
                         "Available implementation types options: %s"
                         % (conf_type, list(iter_type_options)))
    cls = type_map[conf_type]
    assert issubclass(cls, Configurable)
    return cls.from_config(config[conf_type], *args)
