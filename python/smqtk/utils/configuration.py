"""
Helper interface and functions for generalized object configuration, to and from
JSON-compliant dictionaries.

While this interface and utility methods should be general enough to add
JSON-compliant dictionary-based configuration to any object, this was created
in mind with the SMQTK plugin module.

Standard configuration dictionaries should be JSON compliant take the following
general format:

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
Usually, the classes named within a block inherit from a common interface and
the "type" value denotes a selection of a specific sub-class for use, though
this is not required property of these constructs.

"""
import abc
import inspect
import json
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

        This base method is adequate without modification when a class's
        constructor argument types are JSON-compliant.  If one or more are not,
        however, this method then needs to be overridden in order to convert
        from a JSON-compliant stand-in into the more complex object the
        constructor requires.  It is recommended that when complex types *are*
        used they also inherit from the :class:`Configurable` in order to
        hopefully make easier the conversion to and from JSON-compliant
        stand-ins.

        When this method *does* need to be overridden, this usually looks like
        the following pattern:

        .. code-block:: python

           class MyClass (Configurable):

               @classmethod
               def from_config(cls, config_dict, merge_default=True):
                   # Optionally guarantee default values are present in the
                   # configuration dictionary.  This statement pairs with the
                   # ``merge_default=False`` parameter in the super call.
                   # This also in effect shallow copies the given non-dictionary
                   # entries of ``config_dict`` due to the merger with the
                   # default config.
                   if merge_default:
                       config_dict = merge_dict(cls.get_default_config(),
                                                config_dict)

                   #
                   # Perform any overriding here.
                   #

                   # Create and return an instance using the super method.
                   return super(MyClass, cls).from_config(config_dict,
                                                          merge_default=False)

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

        In the most cases, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.  In some cases, where it
        doesn't make sense to store some object constructor parameters are
        expected to be supplied at as configuration values (i.e. must be
        supplied at runtime), this method's returned dictionary may leave those
        parameters out. In such cases, the object's ``from_config``
        class-method would also take additional positional arguments to fill in
        for the parameters that this returned configuration lacks.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """


def make_default_config(configurable_iter):
    """
    Generated default configuration dictionary for the given iterable of
    Configurable-inheriting types.

    For example, assuming the following simple class that descends from
    ``Configurable``, we would expect the following behavior:

    >>> class ExampleConfigurableType (Configurable):
    ...     def __init__(self, a, b):
    ...        ''' Dummy constructor '''
    >>> make_default_config([ExampleConfigurableType]) == {
    ...     'type': None,
    ...     'ExampleConfigurableType': {
    ...         'a': None,
    ...         'b': None,
    ...     }
    ... }
    True

    Note that technically ``ExampleConfigurableType`` is still abstract as it
    does not implement ``get_config``.  The above call to
    ``make_default_config`` still functions because we only use the
    ``get_default_config`` class method and do not instantiate any types given
    to this function.  While functionally acceptable, it is generally not
    recommended to draw configurations from abstract classes.

    :param collections.Iterable[type] configurable_iter:
        An iterable of class types class types that sub-class ``Configurable``.

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


def cls_conf_to_config_dict(cls, conf):
    """
    Helper function for creating the appropriate "standard" smqtk configuration
    dictionary given a `Configurable`-implementing class and a configuration
    for that class.

    This very simple function simply arranges a class, using its __name__
    property, and an associated dictionary into a normal pattern used for
    configuration in SMQTK::

    >>> class SomeClass (object):
    ...     pass
    >>> cls_conf_to_config_dict(SomeClass, {0: 0, 'a': 'b'}) == {
    ...     'type': 'SomeClass',
    ...     'SomeClass': {0: 0, 'a': 'b'}
    ... }
    True

    :param type[Configurable] cls:
        A class type implementing the `Configurable` interface.

    :param dict conf:
        SMQTK standard type-optioned configuration dictionary for the given
        class and dictionary pair.

    :return: "Standard" SMQTK JSON-compliant configuration dictionary
    :rtype: dict

    """
    return {
        "type": cls.__name__,
        cls.__name__: conf
    }


def to_config_dict(c_inst):
    """
    Helper function that transforms the configuration dictionary retrieved from
    ``configurable_inst`` into the "standard" SMQTK configuration dictionary
    format (see above module documentation).

    For example, with a simple DataFileElement:
    >>> from smqtk.representation.data_element.file_element \
            import DataFileElement
    >>> e = DataFileElement(filepath='/path/to/file.txt',
    ...                       readonly=True)
    >>> to_config_dict(e) == {
    ...     "type": "DataFileElement",
    ...     "DataFileElement": {
    ...         "filepath": "/path/to/file.txt",
    ...         "readonly": True,
    ...         "explicit_mimetype": None,
    ...     }
    ... }
    True

    :param Configurable c_inst:
        Instance of a class type that subclasses the ``Configurable`` interface.

    :return: Standard format configuration dictionary.
    :rtype: dict

    """
    c_class = c_inst.__class__
    if isinstance(c_inst, type) or not issubclass(c_class, Configurable):
        raise ValueError("c_inst must be an instance and its type must "
                         "subclass from Configurable. Was given '{}'."
                         .format(type(c_inst)))
    return cls_conf_to_config_dict(c_class, c_inst.get_config())


def cls_conf_from_config_dict(config, type_iter):
    """
    Helper function for getting the appropriate type and configuration
    sub-dictionary based on the provided "standard" SMQTK configuration
    dictionary format (see above module documentation).

    :param dict config:
        Configuration dictionary to draw from.

    :param collections.Iterable[type] type_iter:
        An iterable of class types to select from.

    :raises ValueError:
        This may be raised if:
            - type field not present in ``config``.
            - type field set to ``None``
            - type field did not match any available configuration in the given
              config.
            - Type field did not specify any implementation key.

    :return: Appropriate class type from ``type_iter`` that matches the
        configured type as well as the sub-dictionary from the configuration.
        From this return, ``type.from_config(config)`` should be callable.
    :rtype: (type, dict)
    """
    if 'type' not in config:
        raise ValueError("Configuration dictionary given does not have an "
                         "implementation type specification.")
    conf_type_name = config['type']
    type_map = dict(map(lambda t: (t.__name__, t), type_iter))

    conf_type_options = set(config.keys()) - {'type'}
    # Type provided may either by None, not have a matching block in the
    # config, not have a matching implementation type, or match both.
    if conf_type_name is None:
        raise ValueError("No implementation type specified. Options: %s"
                         % list(conf_type_options))
    elif conf_type_name not in conf_type_options:
        raise ValueError("Implementation type specified as '%s', but no "
                         "configuration block was present for that type. "
                         "Available configuration block options: %s"
                         % (conf_type_name, list(conf_type_options)))
    elif conf_type_name not in type_map:
        raise ValueError("Implementation type specified as '%s', but no "
                         "plugin implementations are available for that type. "
                         "Available implementation types options: %s"
                         % (conf_type_name, list(type_map)))
    cls = type_map[conf_type_name]
    return cls, config[conf_type_name]


def from_config_dict(config, type_iter, *args):
    """
    Helper function for instantiating an instance of a class given the
    configuration dictionary ``config`` from available types provided by
    ``type_iter`` via the ``Configurable`` interface's ``from_config``
    class-method.

    ``args`` are additionally positional arguments to be passed to the type's
    ``from_config`` method on return.

    Example:
    >>> from smqtk.representation import DescriptorElement
    >>> example_config = {
    ...     'type': 'DescriptorMemoryElement',
    ...     'DescriptorMemoryElement': {},
    ... }
    >>> inst = from_config_dict(example_config, DescriptorElement.get_impls(),
    ...                         'type-str', 'some-uuid')
    >>> from smqtk.representation.descriptor_element.local_elements \
            import DescriptorMemoryElement
    >>> isinstance(inst, DescriptorMemoryElement)
    True

    :raises ValueError:
        This may be raised if:
            - type field not present in ``config``.
            - type field set to ``None``
            - type field did not match any available configuration in the given
              config.
            - Type field did not specify any implementation key.
    :raises AssertionError:
        This may be raised if the class specified as the configuration `type`,
        is present in the given ``type_iter`` but is not a subclass of the
        ``Configurable`` interface.
    :raises TypeError: Insufficient/incorrect initialization parameters were
        specified for the specified ``type``'s constructor.

    :param dict config:
        Configuration dictionary to draw from.

    :param collections.Iterable[type] type_iter:
        An iterable of class types to select from.

    :param object args:
        Other positional arguments to pass to the configured class'
        ``from_config`` class method.

    :return: Instance of the configured class type as specified in ``config``
        and as available in ``type_iter``.
    :rtype: smqtk.utils.configuration.Configurable

    """
    cls, cls_conf = cls_conf_from_config_dict(config, type_iter)
    assert issubclass(cls, Configurable), \
        "Configured class type '%s' does not descend from `Configurable`." \
        % cls.__name__
    return cls.from_config(cls_conf, *args)


def configuration_test_helper(inst, config_ignored_params=frozenset(),
                              from_config_args=()):
    """
    Helper function for testing the get_default_config/from_config/get_config
    methods for class types that in part implement the Configurable mixin
    class.  This function also tests that ``inst``'s parent class type's
    ``get_default_config`` returns a dictionary whose keys' match the
    constructor's inspected parameters (except "self" of course).

    This constructs 3 additional instances based on the given instance
    following the pattern::

         inst-1  ->  inst-2  ->  inst-3
                 ->  inst-4

    This refers to ``inst-2`` and ``inst-4`` being constructed from the config
    from ``inst``, and ``inst-3`` being constructed from the config of
    ``inst-2``.  The equivalence of each instance's config is cross-checked
    with the other instances.  This is intended to check that a configuration
    yields the same class configurations and that the config does not get
    mutated by nested instance construction.

    This function uses assert calls to check for consistency.

    We return all instances constructed in case the caller wants to make
    additional instance integrity checks.

    :param Configurable inst:
        Configurable-mixin inheriting class to test.
    :param set[str] config_ignored_params:
        Set of parameter names in the instance type's constructor that are
        ignored by ``get_default_config`` and ``from_config``. This is empty
        by default.
    :param tuple from_config_args:
        Optional additional positional arguments to the input
        ``inst.from_config`` method after the configuration dictionary.
    :returns: Instance 2, 3, and 4 as described above.
    :rtype: (Configurable,Configurable,Configurable)
    """
    assert not isinstance(inst, type), "Passed a type, expected instance."
    inst_T = inst.__class__

    # Patent class default config keys should match constructor keys.
    dflt_cfg = inst_T.get_default_config()
    argspec = inspect.getargspec(inst_T.__init__)
    # Check that keys returned in default config is equivalent to parameters
    # requested by the constructor, minus explicitly provided parameter names
    # to disregard.
    # - Disregarded params are usually for when some arguments are also
    #   required by ``from_config``, i.e. runtime required.
    args_intersect = \
        set(dflt_cfg) == (set(argspec.args[1:]) - config_ignored_params)
    assert args_intersect, \
        "Default configuration dictionary keys does not match the class' " \
        "constructor parameter."
    del args_intersect
    # Should be JSON serializable.
    try:
        assert json.loads(json.dumps(dflt_cfg)) == dflt_cfg, \
            "Default config JSON Serialize -> Deserialize did not match " \
            "original config."
    except TypeError:
        # dumps error
        raise AssertionError("Failed to serialize default config return for "
                             "type {}.".format(inst_T.__name__))
    except ValueError:
        # loads error
        raise AssertionError("Failed to load serialized default config.")

    # Instance config / from_config construction cycle equivalence test.
    # - Checking that configurations are JSON serializable at each step.
    inst_config = inst.get_config()

    # Keys in default and instance configurations should also match.
    assert set(dflt_cfg) == set(inst_config)

    inst2 = inst_T.from_config(inst_config, *from_config_args)
    inst2_config = inst2.get_config()
    inst3 = inst_T.from_config(inst_config, *from_config_args)
    inst3_config = inst3.get_config()
    inst4 = inst_T.from_config(inst2_config, *from_config_args)
    inst4_config = inst4.get_config()
    assert inst_config == inst2_config
    assert inst_config == inst3_config
    assert inst2_config == inst3_config
    assert inst_config == inst4_config
    assert inst3_config == inst4_config

    return inst2, inst3, inst4
