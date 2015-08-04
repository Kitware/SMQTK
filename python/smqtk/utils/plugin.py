# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

Helper methods for higher level plugin module getter methods.

"""

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
                elif (isinstance(classes, collections.Iterable)
                        and not isinstance(classes, basestring)):
                    classes = tuple(classes)
                    log.debug("[%s] Loaded list of %d class types via helper",
                              module_name, len(classes))
                    # check that all class types in iterable are types and
                    # are subclasses of the given base-type
                    for cls in classes:
                        if not (isinstance(cls, type)
                                and cls is not baseclass_type
                                and issubclass(cls, baseclass_type)):
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


def to_config(plugin_getter, arg_skip=0):
    """
    Generated configuration dictionary for the given base-class and associated
    plugin getter method (which returns a dictionary of labels to class types)

    A types parameters, as listed, at the construction parameters for that type.
    Default values are inserted where possible, otherwise None values are
    listed.

    :param plugin_getter: Function that returns a dictionary mapping labels to
        class types.
    :type plugin_getter: () -> dict[str, type]

    :param arg_skip: The number of arguments in a classes constructor to skip
        and not consider as configurable arguments. 0 by default (always skips
        self argument).
    :type arg_skip: int

    :return: Base configuration dictionary with an empty ``type`` field, and
        containing the types and initialization parameter specification for all
        implementation types available from the provided getter method.
    :rtype: dict[str, object]

    """
    d = {"type": None}
    for label, cls in plugin_getter().iteritems():
        d[label] = {}
        argspec = inspect.getargspec(cls.__init__)

        params = argspec.args[1 + arg_skip:]
        num_params = len(params)

        # assuming ``self`` doesn't have a default argument.
        if argspec.defaults:
            vals = ((None,) * (num_params - len(argspec.defaults))
                    + argspec.defaults[-num_params:])
        else:
            vals = (None,) * num_params

        d[label] = dict(zip(params, vals))
    return d


def from_config(config_dict, plugin_getter, *header_args):
    """
    Assuming a configuration dictionary for object specification and
    construction is of the form (as made by the ``to_config`` function):

        {
            "type": "name",
            "name": {
                "param1": "val1",
                "param2": "val2",
                ...
            },
            "other": {
                ...
            },
            ...
        }

    Return an instance of the configured type as found in the given
    ``plugin_getter``.

    :raises KeyError: There was no ``type`` field to inspect, or there was no
        parameter specification for the specified ``type``.
    :raises TypeError: Insufficient/incorrect initialization parameters were
        specified for the specified ``type``'s constructor.

    :param config_dict: Configuration dictionary to draw from.
    :type config_dict: dict[str]

    :param plugin_getter: Function that returns a dictionary mapping labels to
        class types.
    :type plugin_getter: () -> dict[str]

    :param header_args: Positional arguments to be inserted in front of
        configuration parameters when constructing the configured instance.

    :return: Instance of the configured type

    """
    t = config_dict['type']
    cls = plugin_getter()[t]
    try:
        return cls(*header_args, **config_dict[t])
    except TypeError, ex:
        raise TypeError(cls.__name__ + '.' + ex.message)
