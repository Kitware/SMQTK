"""
Helper interface and function for implementing class discovery.

Plugins may be accessed by one of the following ways:
  1. Be defined within SMQTK next to their interface.
  2. The environment variable ``SMQTK_PLUGIN_PATH`` may be set to a number of
     `:`-separated (`;` on Windows) python module paths to where plugin classes
     are defined.
  3. Other installed python packages may define one or more extensions for
     the namespace "smqtk_plugins". This should be a single or list of
     extensions that specify modules within the installed package where
     plugins for export are implemented. Note that we desire modules,
     not objects, for our extensions.

     For example::

         ...
         entry_points={
             "smqtk_plugins": "my_package = my_package.plugins"
         ]
         ...

     Or::

         ...
         entry_points = {
             "smqtk_plugins": [
                 "my_package_mode_1 = my_package.mode_1.plugins",
                 "my_package_mode_2 = my_package.mode_2.plugins",
             ]
         }
         ...

"""

import abc
import collections
import importlib
import inspect
import itertools
import logging
import os
import pkgutil
import queue
import re
import types
from typing import Callable, FrozenSet, Set, Type
import warnings

from stevedore.extension import ExtensionManager

# Template for checking validity of sub-module files
VALID_MODULE_FILE_RE = re.compile(r"^[a-zA-Z]\w*(?:\.py)?$")

# Template for checking validity of module attributes
VALUE_ATTRIBUTE_RE = re.compile(r"^[a-zA-Z]\w*$")

# Environment variable *PATH separator for the current platform.
OS_ENV_PATH_SEP = os.pathsep

EXTENSION_NAMESPACE = "smqtk_plugins"

_EMPTY_FROZENSET_STR: FrozenSet[str] = frozenset()


def _get_local_plugin_modules(log, interface_type, warn=True):
    """
    Get the python modules within the SMQTK tree that are located parallel
    to the module defining the given interface type.

    :param logging.Logger log:
        Logger instance to use for logging.
    :param type interface_type:
        Interface type we want to find modules around.
    :param bool warn:
        If we should warn about module import failures.

    :return: Iterator of python modules parallel to the given interface.
    :rtype: collections.abc.Iterator[types.ModuleType]

    """
    # Get the parent module and the filesystem path to that module.
    # - This is for finding implementations that are defined in the same
    #   parent module as the interface that is inheriting from
    #   ``Pluggable``.
    # - This should do the right thing regardless of whether ``cls`` is
    #   defined in an ``__init__.py`` file or "normal" python module file.
    t_module = inspect.getmodule(interface_type)
    assert t_module is not None, "Interface type module not found."
    # __package__ should correctly get us the parent module path regardless of
    # whether ``interface_type`` is defined in an ``__init__.py`` file or in a
    # parallel python module file.
    t_module_package = t_module.__package__
    assert t_module_package is not None, (
        "Interface type module missing package specification"
    )

    # Containing directory of the module ``interface_class`` is defined in.
    t_module_fp = inspect.getsourcefile(t_module) or inspect.getfile(t_module)
    t_module_dir = os.path.abspath(os.path.dirname(t_module_fp))
    log.debug("Looking for python modules parallel to {} in directory '{}'."
              .format(interface_type.__name__, t_module_dir))

    # Discover sibling modules to interface type's module.
    for importer, module_name, ispackage \
            in pkgutil.iter_modules([t_module_dir]):
        sib_module_path = '.'.join([t_module_package, module_name])
        log.debug("Found sibling module '{}'.".format(sib_module_path))
        try:
            yield importlib.import_module(sib_module_path)
        except Exception as ex:
            if warn:
                warnings.warn("Failed to import module '{}' due to exception: "
                              "({}) {}"
                              .format(sib_module_path, ex.__class__.__name__,
                                      str(ex)))


def _get_envvar_plugin_module(log, env_var, warn=True):
    """
    Get the python modules specified by the given environment variable.

    :param logging.Logger log:
        Logger instance to use for logging.
    :param str env_var:
        Environment variable key to use to look for python module paths to
        load.
    :param bool warn:
        If we should warn about module import failures.

    :return: Iterator of python modules parallel to the given interface.
    :rtype: collections.abc.Iterator[types.ModuleType]

    """
    if env_var in os.environ:
        for p in os.environ[env_var].split(OS_ENV_PATH_SEP):
            # skip empty strings
            if p:
                log.debug("In env variable '{}' found module path '{}'."
                          .format(env_var, p))
                try:
                    yield importlib.import_module(p)
                except Exception as ex:
                    if warn:
                        warnings.warn(
                            "Failed to import module '{}' due to exception: "
                            "({}) {}"
                            .format(p, ex.__class__.__name__, str(ex))
                        )
    else:
        log.debug("No variable '{}' in environment.".format(env_var))


def _get_extension_plugin_modules(log, warn=True):
    """
    Get the modules registered by installed python modules that
    provide extensions in the ``smqtk_plugins`` namespace.

    This function is NOT thread-safe.

    :return: Iterator of python modules registered by installed extensions.
    :rtype: collections.abc.Iterator[types.ModuleType]

    """
    # Get the cached extension manager.
    try:
        m = _get_extension_plugin_modules.ext_manager  # type: ignore
    except AttributeError:
        log.debug("Creating and caching ExtensionManager for namespace '{}'."
                  .format(EXTENSION_NAMESPACE))
        m = ExtensionManager(EXTENSION_NAMESPACE)
        # noinspection PyTypeHints
        _get_extension_plugin_modules.ext_manager = m  # type: ignore
    # Yield registered extensions that are actually modules.
    for ext in m:
        ext_plugin_module = ext.plugin
        if not isinstance(ext_plugin_module, types.ModuleType):
            if warn:
                warnings.warn("Skipping extension provided by package '{} "
                              "({})' that did NOT resolve to a python module "
                              "(got an object of type {} instead: {})."
                              .format(ext.entry_point.dist.key,
                                      ext.entry_point.dist.version,
                                      type(ext_plugin_module).__name__,
                                      ext_plugin_module))
        else:
            yield ext.plugin


def is_valid(cls: Type["Pluggable"], log: logging.Logger, module_path: str, interface_type: Type["Pluggable"]) -> bool:
    """
    Determine if a class type is a valid candidate for plugin discovery.

    In particular, the class type ``cls`` must satisfy several conditions:
    1. It must be a strict subtype of ``interface_type``.
    2. It must not be an abstract class.
    3. It must self-report as usable via its `is_usable()` class method.

    :param Type[Pluggable] cls: The class type whose validity is being tested
    :param logging.Logger log: A logger used to report testing progress
    :param str module_path: The module path containing the definition of ``cls``
    :param Type[Pluggable] interface_type: The base class under consideration

    :return: ``True`` if the class is a valid candidate for discovery, and ``False`` otherwise
    :rtype: bool
    """
    if not isinstance(cls, type):
        # No logging, over verbose, undetermined type.
        return False
    elif cls is interface_type:
        log.debug("[%s.%s] [skip] Literally the base class.",
                  module_path, cls.__name__)
        return False
    elif not issubclass(cls, interface_type):
        log.debug("[%s.%s] [skip] Does not descend from base class.",
                  module_path, cls.__name__)
        return False
    else:
        # We've narrowed cls down to a subclass of `interface_type`.
        # Next we want to know if there are not-yet-implemented
        # abstract methods and, if a Pluggable, it declares itself as
        # "usable."
        cls_abstract_methods: FrozenSet[str] = getattr(
            cls, "__abstractmethods__", _EMPTY_FROZENSET_STR
        )
        # Don't invoke just yet because this may still be abstract
        # until after checking `cls_abstract_methods`.
        cls_is_usable: Callable[[], bool] = getattr(
            cls, "is_usable", lambda: True
        )
        if bool(cls_abstract_methods):
            # Making this a warning as I think this indicates a broken
            # implementation in the ecosystem.
            # noinspection PyUnresolvedReferences
            log.warning('[%s.%s] [skip] Does not implement one or '
                        'more abstract methods: %s',
                        module_path, cls.__name__,
                        list(cls_abstract_methods))
            return False
        elif not cls_is_usable():
            log.debug("[%s.%s] [skip] Class does not report as usable.",
                      module_path, cls.__name__)
            return False
        else:
            log.debug('[%s.%s] [KEEP] Retaining subclass.',
                      module_path, cls.__name__)
            return True


def get_plugins(interface_type, env_var, helper_var,
                warn=True, reload_modules=False, subclasses=False):
    """
    Discover and return classes implementing the given ``interface_type``.

    Discoverable implementations may either be located in sub-modules parallel
    to the definition of the interface class, be located in modules specified in
    the environment variable  ``env_var``, or they may be defined in scope as
    subclasses of ``interface_type``.

    In order to specify additional out-of-scope python modules containing
    interface-class implementations, additions to the given environment variable
    must be made. Entries must be separated by the standard PATH separating
    character based on the operating OS standard (e.g. ';' (for windows) or
    ':' for most everything else). Entries should be importable python module
    paths.

    When looking at module attributes, we only acknowledge those that start with
    an alphanumeric character. '_' prefixed attributes are effectively hidden
    from discovery by this function when merely scanning a module's attributes.

    We require that the base class that we are checking for also descends from
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

    :param bool subclasses:
        Control whether to report on subclasses defined in scope.

    :return: Set of discovered class types descending from type
        ``interface_type`` and ``smqtk.utils.plugin.Pluggable`` whose keys are
        the string names of the class types.
    :rtype: set[type[Pluggable]]

    """
    if not issubclass(interface_type, Pluggable):
        raise ValueError("Required base-class must descend from the Pluggable "
                         "interface!")

    log = logging.getLogger('.'.join([
        __name__, 'get_plugins[{}]'.format(interface_type.__name__)
    ]))

    modules_iter = \
        itertools.chain(
            _get_local_plugin_modules(log, interface_type, warn=warn),
            _get_envvar_plugin_module(log, env_var, warn=warn),
            _get_extension_plugin_modules(log, warn=warn)
        )

    log.debug("Getting plugins for interface '{}'"
              .format(interface_type.__name__))
    class_set: Set[Type[Pluggable]] = set()
    for _module in modules_iter:
        module_path = _module.__name__
        log.debug("Examining module: {}".format(module_path))
        if reload_modules:
            # Invoke reload in case the module changed between imports.
            _module = importlib.reload(_module)
            if _module is None:
                raise RuntimeError("[{}] Failed to reload".format(module_path))

        # Find valid classes in the discovered module by:
        classes = []
        if hasattr(_module, helper_var):
            # Looking for magic variable for import guidance
            classes = getattr(_module, helper_var)
            if classes is None:
                log.debug("[%s] Helper is None-valued, skipping module",
                          module_path)
                classes = []
            elif (isinstance(classes, collections.abc.Iterable) and
                  not isinstance(classes, str)):
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
        class_set.update(cls for cls in classes if is_valid(cls, log, module_path, interface_type))

    # Filter and add all subclasses (not just immediate ones) of the
    # interface_type.
    if subclasses:
        # Use a queue to track the descendant classes of `interface_type`.
        candidates = queue.Queue()

        # Initialize the queue with the immediate subclasses of the target
        # class.
        for class_type in interface_type.__subclasses__():
            candidates.put_nowait(class_type)

        # Continue testing classes from the queue until it is empty.
        try:
            while True:
                # Pull a class off the queue, and keep it if it passes the
                # validation logic. When the queue is empty, that means there's
                # no further work to do, and control will break to the except
                # clause below.
                class_type = candidates.get_nowait()
                if is_valid(class_type, log, class_type.__module__, interface_type):
                    class_set.add(class_type)

                # Whether the class is valid or not, add its subclasses to the
                # queue.
                for subclass in class_type.__subclasses__():
                    candidates.put_nowait(subclass)

        except queue.Empty:
            pass

    return class_set


class NotUsableError (Exception):
    """
    Exception thrown when a pluggable class is constructed but does not report
    as usable.
    """


class Pluggable (metaclass=abc.ABCMeta):
    """
    Interface for classes that have plugin implementations
    """

    __slots__ = ()

    PLUGIN_ENV_VAR = "SMQTK_PLUGIN_PATH"
    PLUGIN_HELPER_VAR = "SMQTK_PLUGIN_CLASS"

    @classmethod
    def get_impls(cls, warn=True, reload_modules=False, subclasses=False):
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

        :param bool subclasses:
            Control whether to report on subclasses defined in scope.

        :return: Set of discovered class types descending from type
            ``interface_type`` and ``smqtk.utils.plugin.Pluggable`` whose keys
            are the string names of the class types.
        :rtype: set[type[Pluggable]]

        """
        # TODO: If reload is False, cache result or use cache
        # TODO: If True, re-cache new result.
        return get_plugins(cls, cls.PLUGIN_ENV_VAR, cls.PLUGIN_HELPER_VAR,
                           warn=warn, reload_modules=reload_modules, subclasses=subclasses)

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
