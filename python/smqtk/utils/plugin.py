"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

Helper interface and functions for higher level plugin module getter methods.

Plugin configuration dictionaries take the following general format:

    {
        "type": "name",
        "impl_name": {
            "param1": "val1",
            "param2": "val2",
            ...
        },
        "other_impl": {
            ...
        },
        ... (etc.)
    }

"""

import abc
import collections
import importlib
import inspect
import logging
import os
import re


valid_module_file_re = re.compile("^[a-zA-Z]\w*(?:\.py)?$")


def get_plugins(base_module, search_dir, helper_var, baseclass_type,
                filter_func=None):
    """
    Discover and return classes found in the given plugin search directory. Keys
    in the returned map are the names of the discovered classes, and the paired
    values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name provided,
    which can either be a single class object or an iterable of class objects,
    to be exported. If the variable is set to None, we skip that module and do
    not import anything. If the variable is not present, we look for a class
    by the same name and casing as the module. If neither are found, the module
    is skipped.

    :param base_module: Base module string path.
    :type base_module: str

    :param search_dir: Directory path to look for modules in.
    :type search_dir: str

    :param helper_var: Name of the expected helper variable.
    :type helper_var: str

    :param baseclass_type: Class type that discovered classes should descend
        from (inherit from).
    :type baseclass_type: type

    :param filter_func: Optional function that, given an imported class, return
        a boolean determining whether this class type should be included in the
        returned map.
    :type filter_func: (type) -> bool

    :return: Map of discovered class object of type ``baseclass_type`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    log = logging.getLogger('.'.join([__name__,
                                      "getPlugins[%s]" % base_module]))
    log.debug("Getting plugins for module '%s'", base_module)
    class_map = {}
    for file_name in os.listdir(search_dir):
        log.debug("Checking file: %s", file_name)
        if valid_module_file_re.match(file_name):
            log.debug("Examining file: %s", file_name)
            module_name = os.path.splitext(file_name)[0]
            # We want any exception this might throw to continue up. If a module
            # in the directory is not importable, the user should know.
            module = importlib.import_module('.%s' % module_name,
                                             package=base_module)
            # Invoke reload in case the module changed between imports.
            module = reload(module)

            # Look for magic variable for import guidance
            classes = []
            if hasattr(module, helper_var):
                classes = getattr(module, helper_var)
                if classes is None:
                    log.debug("[%s] Helper is None, skipping module",
                              module_name)
                elif (isinstance(classes, collections.Iterable) and
                      not isinstance(classes, basestring)):
                    classes = tuple(classes)
                    log.debug("[%s] Loaded list of %d class types via helper",
                              module_name, len(classes))
                    # check that all class types in iterable are types and
                    # are subclasses of the given base-type
                    for cls in classes:
                        if not (isinstance(cls, type) and
                                cls is not baseclass_type and
                                issubclass(cls, baseclass_type)):
                            raise RuntimeError("[%s] Found element in list "
                                               "that is not a class or does "
                                               "not descend from required base "
                                               "class '%s': %s"
                                               % (module_name,
                                                  baseclass_type.__name__,
                                                  cls))
                elif issubclass(classes, baseclass_type):
                    log.debug("[%s] Loaded class type: %s", module_name,
                              classes.__name__)
                    classes = (classes,)
                else:
                    raise RuntimeError("Helper variable set to an invalid "
                                       "value.", module_name)

            # If no helper variable, fall back to finding class by the same name
            # as the module.
            elif hasattr(module, module.__name__):
                classes = getattr(module, module.__name__)
                if issubclass(classes, baseclass_type):
                    log.debug('[%s] Loaded class type by module name: %s',
                              module_name, classes)
                else:
                    raise RuntimeError("[%s] Failed to find valid class by "
                                       "module name fallback. Set helper "
                                       "variable '%s' to None if this module "
                                       "shouldn't provide a %s plugin "
                                       "implementation(s)."
                                       % (module_name, helper_var,
                                          baseclass_type.__name__))

            else:
                log.debug("[%s] Skipping module (no helper variable + no "
                          "module-named class)", module_name)

            for cls in classes:
                if filter_func is None or filter_func(cls):
                    class_map[cls.__name__] = cls
                else:
                    log.debug('[%s] Removed class type "%s" due to filter '
                              'failure.', module_name, cls.__name__)

    return class_map


def make_config(plugin_getter):
    """
    Generated configuration dictionary for the given base-class and associated
    plugin getter method (which returns a dictionary of labels to class types)

    A types parameters, as listed, at the construction parameters for that type.
    Default values are inserted where possible, otherwise None values are
    listed.

    :param plugin_getter: Function that returns a dictionary mapping labels to
        class types.
    :type plugin_getter: () -> dict[str, type]

    :return: Base configuration dictionary with an empty ``type`` field, and
        containing the types and initialization parameter specification for all
        implementation types available from the provided getter method.
    :rtype: dict[str, object]

    """
    d = {"type": None}
    for label, cls in plugin_getter().iteritems():
        # noinspection PyUnresolvedReferences
        d[label] = cls.default_config()
    return d


class ConfigurablePlugin (object):
    """
    Interface for plugin objects that should be configurable via a configuration
    dictionary (think JSON).
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def default_config(cls):
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

    @classmethod
    def from_config(cls, config_dict):
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        """
        # The simple case is that the class doesn't require any special
        # parameters other than those that can be provided via the JSON
        # specification, which we cover here. If an implementation needs
        # something more special, they can override this function.
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


def to_plugin_config(cp_inst):
    """
    Helper method that transforms the configuration dictionary gotten from the
    passed ConfigurablePlugin-subclass instance into the standard multi-plugin
    configuration dictionary format (see above).

    This result of this function would be compatible with being passed to the
    ``from_plugin_config`` function, given the appropriate plugin-getter method.

    TL;DR: This wraps the instance's ``get_config`` return in a certain way.

    :param cp_inst: Instance of a ConfigurablePlugin-subclass.
    :type cp_inst: ConfigurablePlugin

    :return: Plugin-format configuration dictionary.
    :rtype: dict

    """
    name = cp_inst.__class__.__name__
    return {
        "type": name,
        name: cp_inst.get_config()
    }


def from_plugin_config(config, plugin_getter, *args):
    """
    Helper method for instantiating an instance of a class available via the
    provided ``plugin_getter`` function given the plugin configuration
    dictionary ``config``.

    :raises KeyError: There was no ``type`` field to inspect, or there was no
        parameter specification for the specified ``type``.
    :raises TypeError: Insufficient/incorrect initialization parameters were
        specified for the specified ``type``'s constructor.

    :param config: Configuration dictionary to draw from.
    :type config: dict[str]

    :param plugin_getter: Function that returns a dictionary mapping labels to
        class types.
    :type plugin_getter: () -> dict[str, type]

    :param args: Additional argument to be passed to the ``from_config`` method
        on the configured class type.

    :return: Instance of the configured class type as found in the given
        ``plugin_getter``.
    :rtype: ConfigurablePlugin

    """
    t = config['type']
    cls = plugin_getter()[t]
    try:
        # noinspection PyUnresolvedReferences
        return cls.from_config(config[t], *args)
    except TypeError, ex:
        raise TypeError(cls.__name__ + '.' + ex.message)