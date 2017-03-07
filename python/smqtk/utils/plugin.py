"""
Helper interface and functions for higher level plugin module getter methods.

Plugin configuration dictionaries take the following general format:

.. code-block:: json

    {
        "type": "name",
        "impl_name": {
            "param1": "val1",
            "param2": "val2"
        },
        "other_impl": {
            "p1": 4.5,
            "p2": null
        }
    }

"""

import abc
import collections
import importlib
import logging
import os
import re
import sys

import six

try:
    # noinspection PyStatementEffect
    reload
except NameError:
    # noinspection PyUnresolvedReferences
    reload = importlib.reload


# Template for checking validity of sub-module files
VALID_MODULE_FILE_RE = re.compile("^[a-zA-Z]\w*(?:\.py)?$")

# Template for checking validity of module attributes
VALUE_ATTRIBUTE_RE = re.compile("^[a-zA-Z]\w*$")

OS_ENV_PATH_SEP = (sys.platform == "win32" and ';') or ':'


class NotUsableError (Exception):
    """
    Exception thrown when a pluggable class is constructed but does not report
    as usable.
    """


class Pluggable (object):
    """
    Interface for classes that have plugin implementations
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional dependencies
        that may not yet be available on the system, this method should check
        for those dependencies and return a boolean saying if the implementation
        is usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should emit a
                warning detailing why the implementation is not available for
                use.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        raise NotImplementedError("is_usable class-method not implemented for "
                                  "class '%s'" % cls.__name__)

    def __init__(self):
        if not self.is_usable():
            raise NotUsableError("Implementation class '%s' is not currently "
                                 "usable." % self.__class__.__name__)


def get_plugins(base_module_str, internal_dir, dir_env_var, helper_var,
                baseclass_type, warn=True, reload_modules=False):
    """
    Discover and return classes found in the SMQTK internal plugin directory and
    any additional directories specified via an environment variable.

    In order to specify additional out-of-SMQTK python modules containing
    base-class implementations, additions to the given environment variable must
    be made. Entries must be separated by either a ';' (for windows) or ':' (for
    everything else). This is the same as for the PATH environment variable on
    your platform. Entries should be paths to importable modules containing
    attributes for potential import.

    When looking at module attributes, we acknowledge those that start with an
    alphanumeric character ('_' prefixed attributes are hidden from import by
    this function).

    We required that the base class that we are checking for also descends from
    the ``Pluggable`` interface defined above. This allows us to check if a
    loaded class ``is_usable``.

    Within a module we first look for a helper variable by the name provided,
    which can either be a single class object or an iterable of class objects,
    to be specifically exported. If the variable is set to None, we skip that
    module and do not import anything. If the variable is not present, we look
    at attributes defined in that module for classes that descend from the given
    base class type. If none of the above are found, or if an exception occurs,
    the module is skipped.

    :param base_module_str: SMQTK internal string module path in which internal
        plugin modules are located.
    :type base_module_str: str

    :param internal_dir: Directory path to where SMQTK internal plugin modules
        are located.
    :type internal_dir: str

    :param dir_env_var: String name of an environment variable to look for that
        may optionally define additional directory paths to search for modules
        that may implement additional child classes of the base type.
    :type dir_env_var: str

    :param helper_var: Name of the expected module helper attribute.
    :type helper_var: str

    :param baseclass_type: Class type that discovered classes should descend
        from (inherit from).
    :type baseclass_type: type

    :param warn: If we should warn about module import failures.
    :type warn: bool

    :param reload_modules: Explicitly reload discovered modules from source
        instead of taking a potentially cached version of the module.
    :type reload_modules: bool

    :return: Map of discovered class objects descending from type
        ``baseclass_type`` and ``smqtk.utils.plugin.Pluggable`` whose keys are
        the string names of the class types.
    :rtype: dict of (str, type)

    """
    log = logging.getLogger('.'.join([__name__,
                                      "getPlugins[%s]" % base_module_str]))

    if not issubclass(baseclass_type, Pluggable):
        raise ValueError("Required base-class must descend from the Pluggable "
                         "interface!")

    # List of module paths to check for valid sub-classes.
    #: :type: list[str]
    module_paths = []

    # modules nested under internal module
    log.debug("Finding internal modules...")
    for filename in os.listdir(internal_dir):
        if VALID_MODULE_FILE_RE.match(filename):
            module_name = os.path.splitext(filename)[0]
            log.debug("-- %s", module_name)
            module_paths.append('.'.join([base_module_str, module_name]))
    log.debug("Internal modules to search: %s", module_paths)

    # modules from env variable
    log.debug("Extracting env var module paths")
    log.debug("-- path sep: %s", OS_ENV_PATH_SEP)
    if dir_env_var in os.environ:
        env_var_module_paths = os.environ[dir_env_var].split(OS_ENV_PATH_SEP)
        # strip out empty strings
        env_var_module_paths = [p for p in env_var_module_paths if p]
        log.debug("Additional module paths specified in env var: %s",
                  env_var_module_paths)
        module_paths.extend(env_var_module_paths)
    else:
        log.debug("No paths added from environment.")

    log.debug("Getting plugins for module '%s'", base_module_str)
    class_map = {}
    for module_path in module_paths:
        log.debug("Examining module: %s", module_path)
        # We want any exception this might throw to continue up. If a module
        # in the directory is not importable, the user should know.
        try:
            module = importlib.import_module(module_path)
        except Exception as ex:
            if warn:
                log.warn("[%s] Failed to import module due to exception: "
                         "(%s) %s",
                         module_path, ex.__class__.__name__, str(ex))
            continue
        if reload_modules:
            # Invoke reload in case the module changed between imports.
            module = reload(module)
            if module is None:
                raise RuntimeError("[%s] Failed to reload"
                                   % module_path)

        # Find valid classes in the discovered module by:
        classes = []
        if hasattr(module, helper_var):
            # Looking for magic variable for import guidance
            classes = getattr(module, helper_var)
            if classes is None:
                log.debug("[%s] Helper is None-valued, skipping module",
                          module_path)
                classes = []
            elif (isinstance(classes, collections.Iterable) and
                  not isinstance(classes, six.string_types)):
                classes = list(classes)
                log.debug("[%s] Loaded list of %d class types via helper",
                          module_path, len(classes))
            elif issubclass(classes, baseclass_type):
                log.debug("[%s] Loaded class type: %s",
                          module_path, classes.__name__)
                classes = [classes]
            else:
                raise RuntimeError("[%s] Helper variable set to an invalid "
                                   "value: %s" % (module_path, classes))
        else:
            # Scan module valid attributes for classes that descend from the
            # given base-class.
            for attr_name in dir(module):
                if VALUE_ATTRIBUTE_RE.match(attr_name):
                    log.debug("[%s] Checking attribute '%s'", module_path,
                              attr_name)
                    attr = getattr(module, attr_name)
                    # If the attribute looks like a class that descends and
                    # implements the interface, add it to the class list
                    # - we require that base is pluggable, so if class descends
                    #   from the given base-class, it will have
                    #   __abstractmethods__ property.
                    if isinstance(attr, type) and \
                            attr is not baseclass_type and \
                            issubclass(attr, baseclass_type) and \
                            not bool(attr.__abstractmethods__):
                        log.debug("[%s] -- Discovered subclass: %s",
                                  module_path, attr.__name__)
                        classes.append(attr)

        # Check the validity of the discovered class types
        for cls in classes:
            # check that all class types in iterable are types and
            # are subclasses of the given base-type and plugin interface
            if not (isinstance(cls, type) and
                    cls is not baseclass_type and
                    issubclass(cls, baseclass_type)):
                raise RuntimeError("[%s] Found element in list "
                                   "that is not a class or does "
                                   "not descend from required base "
                                   "class '%s': %s"
                                   % (module_path,
                                      baseclass_type.__name__,
                                      cls))
            # Check if the algorithm reports being usable
            elif not cls.is_usable():
                log.debug('[%s] Class type "%s" reported not usable '
                          '(skipping).',
                          module_path, cls.__name__)
            else:
                # Otherwise add it to the output mapping
                class_map[cls.__name__] = cls

    return class_map


def make_config(plugin_map):
    """
    Generated configuration dictionary for the given plugin getter method
    (which returns a dictionary of labels to class types)

    A types parameters, as listed, at the construction parameters for that
    type. Default values are inserted where possible, otherwise None values are
    used.

    :param plugin_map: A dictionary mapping class names to class types.
    :type plugin_map: dict[str, type]

    :return: Base configuration dictionary with an empty ``type`` field, and
        containing the types and initialization parameter specification for all
        implementation types available from the provided getter method.
    :rtype: dict[str, object]

    """
    d = {"type": None}
    for label, cls in six.iteritems(plugin_map):
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


def from_plugin_config(config, plugin_map, *args):
    """
    Helper method for instantiating an instance of a class available via the
    provided ``plugin_getter`` function given the plugin configuration
    dictionary ``config``.

    :raises KeyError: There was no ``type`` field to inspect, or there was no
        parameter specification for the specified ``type``.
    :raises ValueError: Type field did not specify any implementation key.
    :raises TypeError: Insufficient/incorrect initialization parameters were
        specified for the specified ``type``'s constructor.

    :param config: Configuration dictionary to draw from.
    :type config: dict

    :param plugin_map: A dictionary mapping class names to class types.
    :type plugin_map: dict[str, type]

    :param args: Additional argument to be passed to the ``from_config`` method
        on the configured class type.

    :return: Instance of the configured class type as found in the given
        ``plugin_getter``.
    :rtype: smqtk.utils.Configurable

    """
    t = config['type']
    if t is None:
        options = set(config.keys()) - {'type'}
        raise ValueError("No implementation type specified. Options: %s"
                         % options)
    cls = plugin_map[t]
    # noinspection PyUnresolvedReferences
    return cls.from_config(config[t], *args)
