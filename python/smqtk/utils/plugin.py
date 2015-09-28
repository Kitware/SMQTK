"""
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

import collections
import importlib
import logging
import os
import re


valid_module_file_re = re.compile("^[a-zA-Z]\w*(?:\.py)?$")


def get_plugins(base_module, search_dir, helper_var, baseclass_type,
                filter_func=None, reload_modules=False):
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

    :param reload_modules: Explicitly reload discovered modules from source
        instead of taking a potentially cached version of the module.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``baseclass_type`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    log = logging.getLogger('.'.join([__name__,
                                      "getPlugins[%s]" % base_module]))
    log.debug("Getting plugins for module '%s'", base_module)
    class_map = {}
    for file_name in os.listdir(search_dir):
        if valid_module_file_re.match(file_name):
            log.debug("Examining module file: %s", file_name)
            module_name = os.path.splitext(file_name)[0]
            # We want any exception this might throw to continue up. If a module
            # in the directory is not importable, the user should know.
            try:
                module = importlib.import_module('.%s' % module_name,
                                                 package=base_module)
            except Exception, ex:
                log.warn("Failed to import module '%s' due to exception: "
                         "(%s) %s",
                         module_name, ex.__class__.__name__, str(ex))
                continue
            if reload_modules:
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
    Generated configuration dictionary for the given plugin getter method (which
    returns a dictionary of labels to class types)

    A types parameters, as listed, at the construction parameters for that type.
    Default values are inserted where possible, otherwise None values are
    used.

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
        d[label] = cls.get_default_config()
    return d


def to_plugin_config(cp_inst):
    """
    Helper method that transforms the configuration dictionary gotten from the
    passed Configurable-subclass instance into the standard multi-plugin
    configuration dictionary format (see above).

    This result of this function would be compatible with being passed to the
    ``from_plugin_config`` function, given the appropriate plugin-getter method.

    TL;DR: This wraps the instance's ``get_config`` return in a certain way
           that's compatible with ``from_plugin_config``.

    :param cp_inst: Instance of a Configurable-subclass.
    :type cp_inst: Configurable

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
    :type config: dict

    :param plugin_getter: Function that returns a dictionary mapping labels to
        class types.
    :type plugin_getter: () -> dict[str, type]

    :param args: Additional argument to be passed to the ``from_config`` method
        on the configured class type.

    :return: Instance of the configured class type as found in the given
        ``plugin_getter``.
    :rtype: smqtk.utils.Configurable

    """
    t = config['type']
    cls = plugin_getter()[t]
    try:
        # noinspection PyUnresolvedReferences
        return cls.from_config(config[t], *args)
    except TypeError, ex:
        raise TypeError(cls.__name__ + '.' + ex.message)
