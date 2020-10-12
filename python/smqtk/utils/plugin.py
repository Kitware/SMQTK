"""
Helper functions and mixin interface for implementing class type discovery,
filtering and a convenience mixin class.

This package provides a number of `discover_via_...` functions that return sets
of type instances as found by the method described by that function.

These methods may be composed to create a pool of types that may be then
filtered via the `filter_plugin_types` function to those types that are
specifically "plugin types" for the given interface class.
See the `is_valid_plugin` function documentation for what it means to be a
"plugin" of an interface type.

While the above are defined in fairly general terms, the `Pluggable` class type
defined last here is a mixin class that utilizes all of the above in a manner
specific manner for the purposes of SMQTK.
This mixin class defines the class-method ``get_impls()`` that will return
currently discoverable plugins underneath the type it was called on.
This discovery will follow the values of the ``PLUGIN_ENV_VAR`` and
``PLUGIN_NAMESPACE`` class variables defined in the interface class you are
calling ``get_impls()`` from, using inherited values if not immediately
specified.

Because these plugin semantics are pretty low level and commonly utilized,
logging can be extremely verbose. Logging in this module, while still exists,
is set to emit only at log level 1 or lower ("trace").

NOTE: The type annotations for `discover_via_subclasses` and
      `filter_plugin_types` are currently set to the broad `Type` annotation.
      Ideally these should use `Type[T]` instead, but there is currently a
      `known issue with mypy`_ where it aggressively assumes that an annotated
      type *must* be constructable, so it emits an error when the functions are
      called with an abstract `interface_type`. When this is resolved in mypy
      these annotations should be updated.

.. _known issue with mypy: https://github.com/python/mypy/issues/4717

"""

import abc
import importlib
import inspect
import logging
import os
import pkg_resources
import types
from typing import cast, Collection, FrozenSet, Set, Type, TypeVar

# Environment variable *PATH separator for the current platform.
OS_ENV_PATH_SEP = os.pathsep

_EMPTY_FROZENSET_STR: FrozenSet[str] = frozenset()

# Generic Type variable
T = TypeVar("T")
# Type variable for something that would descend from Pluggable
P = TypeVar("P", bound="Pluggable")

LOG = logging.getLogger(__name__)


def is_valid_plugin(cls: Type, interface_type: Type) -> bool:
    """
    Determine if a class type is a valid candidate for plugin discovery.

    In particular, the class type ``cls`` must satisfy several conditions:
    1. It must not literally be the given interface type.
    2. It must be a strict subtype of ``interface_type``.
    3. It must not be an abstract class. (i.e. no lingering abstract methods or
       properties if the `abc.ABCMeta` metaclass has been used).
    4. If the cls is a subclass of Pluggable, it must report as usable via
       its is_usable() class method.

    Logging for this function, when enabled can be very verbose, and is only
    active with a logging level of 1 or lower.

    :param cls: The class type whose validity is being tested
    :param interface_type: The base class under consideration

    :return: ``True`` if the class is a valid candidate for discovery, and
        ``False`` otherwise.
    :rtype: bool
    """
    log_prefix = f"[{interface_type.__name__} ->? {cls.__module__}.{cls.__name__}]"
    llevel = 1
    if cls is interface_type:
        LOG.log(llevel, f"{log_prefix} [skip] Literally the base class.")
        return False
    elif not issubclass(cls, interface_type):
        LOG.log(llevel, f"{log_prefix} [skip] Does not descend from base class.")
        return False
    else:
        # We've narrowed cls down to a subclass of `interface_type`.
        # Next we want to know if there are not-yet-implemented
        # abstract methods.
        if inspect.isabstract(cls):
            # Type checking does not easily introspect that
            # `__abstractmethods__` is an attribute of types derived from that
            # metaclass, thus the use of `getattr` here.
            cls_abstract_methods: FrozenSet[str] = getattr(
                cls, "__abstractmethods__", _EMPTY_FROZENSET_STR
            )
            LOG.log(
                llevel,
                f"{log_prefix} [skip] Does not implement one or "
                f"more abstract methods: {list(cls_abstract_methods)}"
            )
            return False
        elif issubclass(cls, Pluggable) and not cls.is_usable():
            # Class inherits from Pluggable and does not report itself as
            # usable.
            LOG.log(llevel, f"{log_prefix} [skip] Class does not report as usable.")
            return False
        else:
            LOG.log(llevel, f"{log_prefix} [KEEP] Retaining subclass.")
            return True


def _collect_types_in_module(module: types.ModuleType) -> Set[Type]:
    """
    Common method of returning a set of class types defined in a python module.

    If you happened to want to dynamically reload the types in a module that is
    updated during runtime:

    .. code-block:: python

       module = importlib.reload(module)
       _collect_types_in_module(module)
    """
    type_set: Set[Type] = set()
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            type_set.add(attr)
    return type_set


def discover_via_env_var(env_var: str) -> Set[Type]:
    """
    Discover and return types specified in python-importable modules
    specified in the the given environment variable.

    We expect the given environment variable to define zero or more python
    module paths from which to yield all contained type definitions (i.e.
    things that descent from `type`). If there is an empty path element, it is
    skipped (e.g. "foo::bar:baz" will only attempt importing `foo`, `bar` and
    `baz` modules).

    These python module paths should be separated with the same separator as
    would be used in the PYTHONPATH environment variable specification.

    If a module defines no class types, then no types are included from that
    source for return.

    An expected use-case for this discovery method is for modules
    that are not installed but otherwise accessible via the python search path.
    E.g. local modules, modules accessible through PYTHONPATH search path
    modification, modules accessible through `sys.path` modification.

    Any errors raised from attempting to import a module are propagated upward.

    :param env_var: The name of the environment variable to read from.

    :raises ModuleNotFoundError: When one or more module paths specified in the
        given environment variable are not importable.

    :return: Set of discovered types from the modules specified in the
        environment variable's contents.
    """
    type_set: Set[Type] = set()
    env_var_paths = os.environ.get(env_var, "").split(OS_ENV_PATH_SEP)
    llevel = 1
    # If no value, and empty string splits into `[""]`.
    if env_var_paths == [""]:
        LOG.log(
            llevel,
            f"Environment variable `{env_var}` not defined or did not "
            f"contain any module paths."
        )
    for path in env_var_paths:
        # Skip empty strings
        if path:
            # May raise ModuleNotFoundError if `path` is not a valid,
            # importable module path.
            m = importlib.import_module(path)
            m_tset = _collect_types_in_module(m)
            LOG.log(
                llevel,
                f"For environment variable `{env_var}`, found module "
                f"path `{path}` with types: "
                f"{[t.__name__ for t in m_tset]}"
            )
            type_set.update(m_tset)
    return type_set


class NotAModuleError(Exception):
    """
    Exception for when the `discover_via_entrypoint_extensions` function
    found an entrypoint that was *not* a module specification.
    """


def discover_via_entrypoint_extensions(entrypoint_ns: str) -> Set[Type]:
    """
    Discover and return types defined in modules exposed through the
    entry-point extensions defined for the given namespace by installed python
    packages.

    Other installed python packages may define one or more extensions for
    a namespace, as specified by `ns`, in their "setup.py". This should be a
    single or list of extensions that specify modules within the installed
    package where plugins for export are implemented.

    Currently, this method only accepts extensions that export a module as
    opposed to specifications of a specific attribute in a module.
    This is due to other methods of type discovery not necessarily honoring the
    selectivity that specific attribute specification provides
    (Looking at you `__subclasses__`...).

    For example, as a single specification string::

        ...
        entry_points={
            "smqtk_plugins": "my_package = my_package.plugins"
        ]
        ...

    Or in list form of multiple specification strings::

        ...
        entry_points = {
            "smqtk_plugins": [
                "my_package_mode_1 = my_package.mode_1.plugins",
                "my_package_mode_2 = my_package.mode_2.plugins",
            ]
        }
        ...

    :param entrypoint_ns: The name of the entry-point mapping in  to look for
        extensions under.

    :return: Set of discovered types from the modules and class types specified
        in the extensions under the specified entry-point.
    """
    type_set: Set[Type] = set()
    for entry_point in pkg_resources.iter_entry_points(
        entrypoint_ns
    ):  # type: pkg_resources.EntryPoint
        m = entry_point.load()
        if not isinstance(m, types.ModuleType):
            ep_dist = entry_point.dist
            ep_dist_key = getattr(ep_dist, "key", "UNKNOWN-PACKAGE")
            ep_dist_ver = getattr(ep_dist, "version", "UNKNOWN-VERSION")
            raise NotAModuleError(
                f"Extension provided by the package '{ep_dist_key} "
                f"(version: {ep_dist_ver})' did NOT resolve to a python "
                f"module (got an object of type {type(m).__name__} instead: "
                f"{m})."
            )
        else:
            type_set.update(_collect_types_in_module(m))
    return type_set


def discover_via_subclasses(interface_type: Type) -> Set[Type]:
    """
    Utilize the ``__subclasses__`` to discover nested subclasses for a given
    interface type.

    This approach will be able to observe any implementations that have been
    defined, anywhere at all, at the point of invocation, which can circumvent
    efforts towards specificity that other discovery methods may provide. For
    example, `discover_via_entrypoint_extensions` may return a single type
    that was specifically exported from a module whereas this method will,
    called afterwards, yield all the other types defined in that
    entry-point-imported module.

    The use of this discovery method may also result in different returns
    depending on the import state at the time of invocation. E.g. further
    imports may increase the quantity of returns from this function.

    This function uses depth-first-search when traversing sub-class tree.

    Reference:
      https://docs.python.org/3/library/stdtypes.html#class.__subclasses__

    NOTE: subclasses are retained via weak-references, so if a normal condition
          is exposing types from something that otherwise raised an exception
          or if a local definition is leaking, apparently an `import gc;
          gc.collect()` wipes out the return as long as it's not referenced, of
          course as long as its reference is not retained by something.

    :param interface_type: The interface type to recursively find sub-classes
        under.
    :return: Set of recursive subclass types under `interface_type`.
    """
    # __subclasses__ only returns *immediate* subclasses, i.e. one level.
    # To get nested subclasses we'll have to do some graph traversal.
    class_set = set()

    # Use a list (stack behavior) to track the descendant classes of
    # `interface_type`. Depth- vs. Breadth-first search should not matter here,
    # so just using just using lists here for theoretically more optimal array
    # caching.
    candidates = interface_type.__subclasses__()
    while candidates:
        class_type = candidates.pop()
        class_set.add(class_type)
        candidates.extend(class_type.__subclasses__())
    return class_set


def filter_plugin_types(
    interface_type: Type, candidate_pool: Collection[Type]
) -> Set[Type]:
    """
    Filter the given set of types to those that are "plugins" of the given
    interface type.

    See the documentation for :py:func:`is_valid_plugin` for what we define a
    "plugin type" to be relative to the given `interface_type`.

    We consider that there may be duplicate type instances in the given
    candidate pool. Due to this we will consider an instance of a type only
    once and return a set type to contain the validated types.

    :param interface_type: The parent type to filter on.
    :param candidate_pool: Some iterable of types from which to collect
        interface type plugins from.
    :return: Set of types that are considered "plugins" of the interface types
        following the above listed rules.
    """
    return {cls for cls in candidate_pool if is_valid_plugin(cls, interface_type)}


class NotUsableError(Exception):
    """
    Exception thrown when a pluggable class is constructed but does not report
    as usable.
    """


class Pluggable(metaclass=abc.ABCMeta):
    """
    Interface for classes that have plugin implementations
    """

    __slots__ = ()

    PLUGIN_ENV_VAR = "SMQTK_PLUGIN_PATH"
    PLUGIN_NAMESPACE = "smqtk_plugins"

    @classmethod
    def get_impls(cls: Type[P]) -> Set[Type[P]]:
        """
        Discover and return a set of classes that implement the calling class.

        See the ``get_plugins`` function for more details on the logic of how
        implementing classes (aka "plugins") are discovered.

        The class-level variables ``PLUGIN_ENV_VAR`` and ``PLUGIN_HELPER_VAR``
        may be overridden to change what environment and helper variable are
        looked for, respectively.

        :return: Set of discovered class types that are considered "valid"
            plugins of this type. See :py:func:`is_valid_plugin` for what we
            define a "valid" type to be be relative to this class.

        """
        candidate_types = {
            *discover_via_env_var(cls.PLUGIN_ENV_VAR),
            *discover_via_entrypoint_extensions(cls.PLUGIN_NAMESPACE),
            *discover_via_subclasses(cls)
        }
        resolved_types = cast(
            Set[Type[P]],
            filter_plugin_types(cls, candidate_types)
        )
        return resolved_types

    @classmethod
    def is_usable(cls) -> bool:
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional
        dependencies that may not yet be available on the system, or other
        runtime conditions, this method may be overridden to check for those
        and return a boolean saying if the implementation is available for
        usable. When this method returns `True`, the class is declaring that it
        should be constructable and usable in the current environment.

        By default, this method will return True unless a sub-class overrides
        this class-method with their specific logic.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should emit a
                (user) warning, or some other kind of logging, detailing why
                the implementation is not available for use.

        :return: Boolean determination of whether this implementation is
            usable in the current environment.
        :rtype: bool

        """
        return True

    def __init__(self) -> None:
        if not self.is_usable():
            raise NotUsableError(
                "Implementation class '%s' is not currently "
                "usable." % self.__class__.__name__
            )
