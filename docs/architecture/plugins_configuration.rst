Plugins and Configuration
-------------------------
SMQTK provides plugin and configuration utilities to support the creation of
interface classes that have a convenient means of
accessing implementing types, paired ability to dynamically instantiate
interface implementations based on a configuration derived by constructor
introspection.

While these two primary mixin classes function independently and can be
utilized on their own, their combination is symbiotic and allows for users of
derivative interfaces to create tools in terms of the interfaces and leave the
specific selection of implementations for configuration time.

Later, we will introduce the two categories of configurable and
(usually) pluggable class classes found within SMQTK.


The :class:`~smqtk.utils.plugin.Pluggable` Mixin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Motivation:**
We want to be able to define interfaces to generic concepts and structures
that higher level tools can be defined around without strictly catering
themselves to any particular implementation, while additionally allowing
freedom in implementation variety without overly restricting implementations.

In SMQTK, this is addressed via the :class:`~smqtk.utils.plugin.Pluggable`
abstract mixin class:

.. code-block:: python

   import abc
   from smqtk.utils.plugin import Pluggable

   class MyInterface(Pluggable):

       @abc.abstractmethod
       def my_behavior(self, x: str) -> int:
           """My fancy behavior."""

   if __name__ == "__main__":
       # Discover currently available implementations and print out their names
       impl_types = MyInterface.get_impls()
       print("MyInterface implementations:")
       for t in impl_types:
           print(f"- {t.__name__}")


Interfaces and Implementations
""""""""""""""""""""""""""""""
Classes that inherit from the :class:`~smqtk.utils.plugin.Pluggable`
mixin are considered either pluggable interfaces or plugin implementations
depending on whether they fully implement abstract methods.

Interface implementations bundled within SMQTK are generally defined alongside
their parent interfaces.
However, other sources, e.g. other python packages, may expose their own plugin
implementations via setting a system environment variable or via python
extensions.


The :class:`~smqtk.utils.configuration.Configurable` Mixin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Motivation:**
We want generic helpers to enable serializable configuration for classes while
minimally impacting standard class development.

SMQTK provides the :class:`~smqtk.utils.configuration.Configurable` mixin class
as well as other helper utility functions in :mod:`smqtk.utils.configuration`
for generating class instances from configurations.
These use python's :mod:`inspect` module to determine constructor
parameterization and default configurations.

Currently this module uses the JSON-serializable format as the basis for input
and output configuration dictionaries as a means of defining a relatively
simple playing field for communication.
Serialization and deserialization is detached from these configuration
utilities so tools may make their own decisions there.
Python dictionaries are used as a medium in between serialization and
configuration input/output.

Classes that inherit from :class:`~smqtk.utils.configuration.Configurable` *do*
need to at a minimum implement the
:meth:`~smqtk.utils.configuration.Configurable.get_config` instance method.


Algorithms and Representations - The Combination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Interfaces found in SMQTK are generally binned into two categories:
representations and algorithms.

Algorithms are interfaces to some function or operation, specifically
parameterized through their constructor and generally parameterized via the
algorithm's interface.
The :class:`~smqtk.algorithms.SmqtkAlgorithm` base class inherits from both
:class:`~smqtk.utils.plugin.Pluggable` and
:class:`~smqtk.utils.configuration.Configurable` mixins so that all descendents
gain access to the synergy they provide.
These are located under the :mod:`smqtk.algorithms` sub-module.

Representations are interfaces to structures that are intended to specifically
store some sort of data structure.
Currently, the :class:`~smqtk.representation.SmqtkRepresentation` only inherits
directly from :class:`~smqtk.utils.configuration.Configurable`, as there are
some representational structures which desire configurability but to which
variable implementations do not make sense (like
:class:`~smqtk.representation.DescriptorElementFactory`).
However most sub-classes do additionally inherit from
:class:`~smqtk.utils.plugin.Pluggable` (like
:class:`~smqtk.representation.DescriptorElement`).
These are located under the :mod:`smqtk.representation` sub-module.


Implementing a Pluggable Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following are examples of how to add and expose new plugin implementations
for existing algorithm and representation interfaces.

SMQTK's plugin discovery via the :meth:`~smqtk.utils.plugin.Pluggable.get_impls`
method currently allows for finding a plugin implementations in 3 ways:

* sub-classes of an interface type defined in the current runtime.

* within python modules listed in the environment variable specified by
  ``YourInterface.PLUGIN_ENV_VAR``. (default SMQTK environment variable name
  is ``SMQTK_PLUGIN_PATH``, which is defined in
  :attr:`Pluggable.PLUGIN_ENV_VAR`).

* within python modules specified under the entry point extensions namespace
  defined by ``YourInterface.PLUGIN_NAMESPACE`` (default SMQTK extension
  namespace is ``smqtk_plugins``, which is defined in
  :attr:`Pluggable.PLUGIN_NAMESPACE`).

Within SMQTK
""""""""""""
A new interface implementation within the SMQTK source-tree is generally
implemented or exposed parallel to where the parent interface is defined.
This has been purely for organizational purposes.
Once we define our implementation, we will then expose that type in an existing
module that is already referenced in SMQTK's list of entry point extensions.

In this example, we will show how to create a new implementation for the
:class:`~smqtk.algorithms.classifier.Classifier` algorithm
interface.
This interface is defined within SMQTK at, from the root of the source tree,
:file:`python/smqtk/algorithms/classifier/_interface_classifier.py`.
We will create a new file, :file:`some_impl.py`, that will be placed in the
same directory.

We'll define our new class, lets call it ``SomeImpl``, in a file
:file:`some_impl.py`::

    python/
    └── smqtk/
        └── algorithms/
            └── classifier/
                ├── _interface_classifier.py
                ├── some_impl.py     # new
                └── ...

In this file we will need to define the :class:`SomeImpl` class and all parent
class abstract methods in order for the class to satisfy the definition of an
"implementation":

.. code-block:: python

    from smqtk.algorithms import Classifier

    class SomeImpl (Classifier):
        """
        Some documentation for this specific implementation.
        """

        # Our implementation-specific constructor.
        def __init__(self, paramA=1, paramB=2):
            ...

        # Abstract methods from Configurable.
        # (Classifier -> SmqtkAlgorithm -> Configurable)
        def get_config(self):
            # As per Configurable documentation, this returns the same non-self
            # keys as the constructor.
            return {
                "paramA": ...,
                "paramB": ...,
            }

        # Classifier's abstract methods.
        def get_labels(self):
            ...

        def _classify_arrays(self, array_iter):
            ...

The final step to making this implementation discoverable is to add an
import of this class to the existing hub of classifier plugins in
:file:`python/smqtk/algorithms/classifier/_plugins.py`:

.. code-block:: python

   ...
   from .some_impl import SomeImpl

With all abstract methods defined, this implementation will now be included
in the returned set of implementation types for the parent
:class:`~smqtk.algorithms.classifier.Classifier` interface:

.. code-block:: python

    >>> from smqtk.algorithms import Classifier
    >>> Classifier.get_impls()
    set([..., smqtk.algorithms.classifier.some_impl.SomeImpl, ...])

:class:`SomeImpl` above should also be all set for configuration because it
defines the one required abstract method
:meth:`~smqtk.utils.configuration.Configurable.get_config` and because
its constructor is only anticipating JSON-compliant types.
If more complicated types are desired by the constructor the additional methods
would need to be overridden/extended as defined in the
:mod:`smqtk.utils.configuration` module.

Within another python package
"""""""""""""""""""""""""""""
When implementing a pluggable interface in another python package, the proper
method of export is via a package's entry point specifications using the
namespace key defined by the parent interface (by default the ``smqtk_plugins``
value is defined by :attr:`smqtk.utils.plugin.Pluggable.PLUGIN_NAMESPACE`).

For example, let's assume that a separate python package, ``OtherPackage``
we'll call it, defines a
:class:`~smqtk.algorithms.classifier.Classifier`-implementing sub-class
:class:`OtherClassifier` in the module :mod:`OtherPackage.other_classifier`.
This module location can be exposed via the package's :file:`setup.py`
entry points metadata, using the ``smqtk_plugins`` key, like the following:

.. code-block:: python

    from setuptools import setup

    ...

    setup(
        ...
        entry_points={
            'smqtk_plugins': 'my_plugins = OtherPackage.other_classifier'
        }
    )

If this other package had multiple sub-modules in which SMQTK plugins were
defined, the ``smqtk_plugins`` entry value may instead be a list:

.. code-block:: python

    setup(
        ...
        entry_points={
            'smqtk_plugins': [
                'classifier_plugins = OtherPackage.other_classifier',
                'other_plugins = OtherPackage.other_plugins',
            ]
        }
    )


Reference
^^^^^^^^^

:mod:`smqtk.utils.plugin`
"""""""""""""""""""""""""
.. automodule:: smqtk.utils.plugin
   :members:

:mod:`smqtk.utils.configuration`
""""""""""""""""""""""""""""""""
.. automodule:: smqtk.utils.configuration
   :members:


Reload Use Warning
''''''''''''''''''
While the :func:`smqtk.utils.plugin.get_plugins` function allows for reloading
discovered modules for potentially new content, this is not recommended under
normal conditions.
When reloading a plugin module after :mod:`pickle` serializing an instance of
an implementation, deserialization causes an error because the original class
type that was pickled is no longer valid as the reloaded module overwrote the
previous plugin class type.
