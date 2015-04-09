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
import logging
import os
import re


valid_module_file_re = re.compile("^[a-zA-Z].*(?:\.py)?$")


def get_plugins(base_module, search_dir, helper_var, baseclass_type):
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

    :return: Map of discovered class object of type ``baseclass_type`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    log = logging.getLogger("getPlugins[%s]" % base_module)
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
                    log.debug("[%s] Loaded list of %d classes via helper",
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
                                               "class '%s': %s",
                                               module_name,
                                               baseclass_type.__name__,
                                               cls)
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
                                       "implementation(s).",
                                       module_name, helper_var,
                                       baseclass_type.__name__)

            else:
                log.debug("[%s] Skipping module (no helper variable + no "
                          "module-named class)", module_name)

            for cls in classes:
                class_map[cls.__name__] = cls

    return class_map
