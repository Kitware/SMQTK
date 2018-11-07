"""
Helper interface and function for implementing class discovery.
"""

import abc
import collections
import importlib
import inspect
import logging
import os
import pkgutil
import re

import six
from six.moves import reload_module


# Template for checking validity of sub-module files
VALID_MODULE_FILE_RE = re.compile("^[a-zA-Z]\w*(?:\.py)?$")

# Template for checking validity of module attributes
VALUE_ATTRIBUTE_RE = re.compile("^[a-zA-Z]\w*$")

# Environment variable *PATH separator for the current platform.
OS_ENV_PATH_SEP = os.pathsep


class NotUsableError (Exception):
    """
    Exception thrown when a pluggable class is constructed but does not report
    as usable.
    """


@six.add_metaclass(abc.ABCMeta)
class Pluggable (object):
    """
    Interface for classes that have plugin implementations
    """

    PLUGIN_ENV_VAR = "SMQTK_PLUGIN_PATH"
    PLUGIN_HELPER_VAR = "SMQTK_PLUGIN_CLASS"

    @classmethod
    def get_impls(cls, warn=True, reload_modules=False):
        """
        Discover and return a set of classes that implement the calling class.

        See the ``get_plugins`` function for more details on the logic of how
        implementing classes (aka "plugins") are discovered.

        The class-level variables ``PLUGIN_ENV_VAR`` and ``PLUGIN_HELPER_VAR``
        may be overridden to change what environment and helper variable are
        looked for, respectively.

        :param bool warn:
        If we should warn about module import failures.

        :param bool reload_modules:
            Explicitly reload discovered modules from source.

        :return: Set of discovered class types descending from type
            ``interface_type`` and ``smqtk.utils.plugin.Pluggable`` whose keys are
            the string names of the class types.
        :rtype: set[type[Pluggable]]

        """
        # TODO: If reload is False, cache result or use cache
        # TODO: If True, re-cache new result.
        return get_plugins(cls, cls.PLUGIN_ENV_VAR, cls.PLUGIN_HELPER_VAR,
                           warn=warn, reload_modules=reload_modules)

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


def get_plugins(interface_type, env_var, helper_var,
                warn=True, reload_modules=False):
    """
    Discover and return classes implementing the given ``interface_class``.

    Discoverable implementations may either be located in sub-modules parallel
    to the definition of the interface class or be located in modules specified
    in the environment variable  ``env_var``.

    In order to specify additional out-of-scope python modules containing
    interface-class implementations, additions to the given environment variable
    must be made. Entries must be separated by the standard PATH separating
    character based on the operating OS standard (e.g. ';' (for windows) or
    ':' for most everything else). Entries should be importable python module
    paths.

    When looking at module attributes, we only acknowledge those that start with
    an alphanumeric character. '_' prefixed attributes are effectively hidden
    from discovery by this function when merely scanning a module's attributes.

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

    :param type interface_type:
        Interface class type of which we want to discover implementations of
        (the plugins).

    :param str env_var:
        String name of an environment variable defining additional python module
        paths, whose child modules are searched for implementing sub-classes of
        the specified base type.

    :param str helper_var:
        Name of the expected module helper attribute.

    :param bool warn:
        If we should warn about module import failures.

    :param bool reload_modules:
        Explicitly reload discovered modules from source instead of taking a
        potentially cached version of the module.

    :return: Set of discovered class types descending from type
        ``interface_type`` and ``smqtk.utils.plugin.Pluggable`` whose keys are
        the string names of the class types.
    :rtype: set[type[Pluggable]]

    """
    if not issubclass(interface_type, Pluggable):
        raise ValueError("Required base-class must descend from the Pluggable "
                         "interface!")

    # Get the parent module and the filesystem path to that module.
    # - This is for finding implementations that are defined in the same
    #   parent module as the interface that is inheriting from
    #   ``Pluggable``.
    # - This should do the right thing regardless of whether ``cls`` is
    #   defined in an ``__init__.py`` file or "normal" python module file.
    interface_type_module = inspect.getmodule(interface_type)
    # __package__ should correctly get us the parent module regardless of
    # whether ``interface_class`` is defined in an ``__init__.py`` file or in a
    # parallel python module file.
    interface_type_module_package = interface_type_module.__package__

    log = logging.getLogger('.'.join(
        [__name__, "getPlugins[%s]" % interface_type_module_package]
    ))

    # List of module paths to check for valid sub-classes.
    #: :type: list[str]
    module_paths = []

    # - Modules parallel to module defining the given ``interface_type``
    log.debug("Finding parallel modules...")
    # Containing directory of the module ``interface_class`` is defined in.
    interface_type_module_dir = os.path.abspath(
        os.path.dirname(inspect.getfile(interface_type_module))
    )
    for importer, module_name, ispackage \
            in pkgutil.iter_modules([interface_type_module_dir]):
        module_paths.append('.'.join([interface_type_module_package,
                                      module_name]))
    log.debug("Parallel modules to search: %s", module_paths)

    # - Modules from env variable
    log.debug("Extracting env var module paths")
    log.debug("-- path sep: %s", OS_ENV_PATH_SEP)
    if env_var in os.environ:
        env_var_module_paths = os.environ[env_var].split(OS_ENV_PATH_SEP)
        # strip out empty strings
        env_var_module_paths = [p for p in env_var_module_paths if p]
        log.debug("Additional module paths specified in env var: %s",
                  env_var_module_paths)
        module_paths.extend(env_var_module_paths)
    else:
        log.debug("No module paths added from environment.")

    log.debug("Getting plugins for interface '%s'", interface_type.__name__)
    class_set = set()
    for module_path in module_paths:
        log.debug("Examining module: %s", module_path)
        # We want any exception this might throw to continue up. If a module
        # in the directory is not importable, the user should know.
        try:
            _module = importlib.import_module(module_path)
        except Exception as ex:
            if warn:
                log.warn("[%s] Failed to import module due to exception: "
                         "(%s) %s",
                         module_path, ex.__class__.__name__, str(ex))
            continue
        if reload_modules:
            # Invoke reload in case the module changed between imports.
            # six should find the right thing.
            # noinspection PyCompatibility
            _module = reload_module(_module)
            if _module is None:
                raise RuntimeError("[%s] Failed to reload"
                                   % module_path)

        # Find valid classes in the discovered module by:
        classes = []
        if hasattr(_module, helper_var):
            # Looking for magic variable for import guidance
            classes = getattr(_module, helper_var)
            if classes is None:
                log.debug("[%s] Helper is None-valued, skipping module",
                          module_path)
                classes = []
            elif (isinstance(classes, collections.Iterable) and
                  not isinstance(classes, six.string_types)):
                classes = list(classes)
                log.debug("[%s] Loaded list of %d class types via helper",
                          module_path, len(classes))
            # Thus, non-iterable value.
            elif isinstance(classes, type) \
                    and issubclass(classes, interface_type):
                log.debug("[%s] Loaded class type: %s",
                          module_path, classes.__name__)
                classes = [classes]
            else:
                raise RuntimeError("[%s] Helper variable set to an invalid "
                                   "value: %s" % (module_path, classes))
        else:
            # Scan module valid attributes for classes that descend from the
            # given base-class.
            log.debug("[%s] No helper, scanning module attributes",
                      module_path)
            for attr_name in dir(_module):
                if VALUE_ATTRIBUTE_RE.match(attr_name):
                    classes.append(getattr(_module, attr_name))

        # Check the validity of the discovered class types in this module.
        for cls in classes:
            # check that all class types in iterable are:
            # - Class types,
            # - Subclasses of the given base-type and plugin interface
            # - Not missing any abstract implementations.
            #
            # noinspection PyUnresolvedReferences
            if not isinstance(cls, type):
                # No logging, over verbose, undetermined type.
                pass
            elif cls is interface_type:
                log.debug("[%s.%s] [skip] Literally the base class.",
                          module_path, cls.__name__)
            elif not issubclass(cls, interface_type):
                log.debug("[%s.%s] [skip] Does not descend from base class.",
                          module_path, cls.__name__)
            elif bool(cls.__abstractmethods__):
                # Making this a warning as I think this indicates a broken
                # implementation in the ecosystem.
                # noinspection PyUnresolvedReferences
                log.warn('[%s.%s] [skip] Does not implement one or more '
                         'abstract methods: %s',
                         module_path, cls.__name__,
                         list(cls.__abstractmethods__))
            elif not cls.is_usable():
                log.debug("[%s.%s] [skip] Class does not report as usable.",
                          module_path, cls.__name__)
            else:
                log.debug('[%s.%s] [KEEP] Retaining subclass.',
                          module_path, cls.__name__)
                class_set.add(cls)

    return class_set
