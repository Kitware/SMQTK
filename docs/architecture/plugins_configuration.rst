Plugins and Configuration
-------------------------
SMQTK provides generic plugin and introspective configuration mixin classes to
support interface implementation discovery and their translation form/to JSON
as a plain-text configuration format.

While these two mixins function independently and can be utilized on their own,
their combination is symbiotic and allows for users of SMQTK algorithms and
representations to create tools in terms of interfaces and leave the specific
selection of implementations for configuration time.

Later, we will introduce the two categories of configurable and
(usually) pluggable class classes found within SMQTK.


Plugins
^^^^^^^
**Motivation:**
We want to be able to define interfaces to generic concepts and structures
around which higher order tools can be defined without strictly catering
themselves to any particular implementation, while additionally allowing
freedom in implementation variety without overly restricting implementation
location.

In SMQTK, this is addressed via the :meth:`~smqtk.utils.plugin.get_plugins`
function and the :class:`~smqtk.utils.plugin.Pluggable` abstract mixin class.


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
for generating, and producing class instances from, configurations.
These use python's :mod:`introspect` module to determine default
configurations.

Currently this module deals in JSON for input and output configuration.
Python dictionaries are used as a medium in between serialization and class
input/output.

Classes that inherit from :class:`~smqtk.utils.configuration.Configurable` *do*
need to at a minimum implement the
:meth:`~smqtk.utils.configuration.Configurable.get_config` instance method.
This does detract from the "minimal impact" intent of this mixin, but other
methods of allowing introspection of internal parameters require additional
structural components in the parent/implementing class.


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

SMQTK's plugin discovery allows for exposure of plugin implementations in 3
ways:
  * Parallel in location to parent interface.
  * Python module path of implementation model included in the
    ``SMQTK_PLUGIN_PATH`` environment variable (see reference for formatting).
  * An entrypoint in a python package's ``setup.py``.

Within SMQTK
""""""""""""
A new interface implementation within the SMQTK source-tree is generally
implemented or exposed parallel to where the parent interface is defined.

As an example, we will show how to create a new implementation for the
:class:`~smqtk.algorithms.classifier.Classifier` algorithm
interface.
This interface is defined within SMQTK at, from the root of the source tree,
:file:`python/smqtk/algorithms/classifier/_interface_classifier.py`.
We will create a new file, :file:`some_impl.py`, that will be placed in the
same directory with the intention that our new plugin will be picked up based
on parallel locality to the parent interface class.

We'll define our new class, lets call it ``SomeImpl``, in a file
:file:`some_impl.py`::

    python/
    └── smqtk/
        └── algorithms/
            └── classifier/
                ├── ...
                ├── _interface_classifier.py
                ├── some_impl.py     # new

In this file we will need to define the :class:`SomeImpl` class and all parent
class abstract methods in order for the class to satisfy the definition of an
"implementation":

.. code-block:: python

    from smqtk.algorithms import Classifier

    class SomeImpl (Classifier):
        """
        Some documentation for this specific implementation.
        """

        # Abstract methods from Pluggable.
        # (Classifier -> SmqtkAlgorithm -> Pluggable)
        @classmethod
        def is_usable(cls):
            ...

        # Our implementation-specific constructor.
        def __init__(self, paramA=1, paramB=2):
            ...

        # Abstract methods from Configurable.
        # (Classifier -> SmqtkAlgorithm -> Configurable)
        def get_config(self):
            return {
                "paramA": ...,
                "paramB": ...,
            }

        # Classifier's abstract methods.
        def get_labels(self):
            ...

        def _classify_arrays(self, array_iter):
            ...

With all abstract methods defined, this implementation should now be included
in the returned set of implementation types for the parent
:class:`~smqtk.algorithms.classifier.Classifier` interface:

.. code-block:: python

    >>> from smqtk.algorithms import Classifier
    >>> Classifier.get_impls()
    set([..., SomeImpl, ...])

:class:`SomeImpl` above should also be all set for configuration because of it
defining :meth:`~smqtk.utils.configuration.Configurable.get_config` and because
it's constructor is only anticipating JSON-conpliant types.
If more complicated types are desired by the constructor the additional methods
would need to be overriden/extended as defined in the
:mod:`smqtk.utils.configuration` module.

More Complicated Implementations
''''''''''''''''''''''''''''''''
Interface-parallel implementation discovery also allows for nested sub-modules.
This is useful when an implementation requires specific or extensive support
utilities.
The :file:`__init__.py` file of the sub-module should at least expose concrete
implementation classes that should be exported as attributes for the plugin
discovery to find.
For example, such a nested sub-module implementation might look like the
following on the filesystem::

    python/
    └── smqtk/
        └── algorithms/
            └── classifier/
                ├── ...
                ├── some_impl.py     # from above
                └── other_impl/      # new
                    └── __init__.py  # new

Within another python package
"""""""""""""""""""""""""""""
When implementing a pluggable interface in another python package, the proper
method of export is via a package's entrypoint specifications using the
``smqtk_plugins`` key.

For example, let's assume that a separate python package, ``OtherPackage``
we'll call it, defines a
:class:`~smqtk.algorithms.classifier.Classifier`-implementing sub-class
:class:`OtherClassifier` in the module :mod:`OtherPackage.other_classifier`.
This module location can be exposed via the package's :file:`setup.py`
entrypoints metadata, using the ``smqtk_plugins`` key, like the following:

.. code-block:: python

    from setuptools import setup

    ...

    setup(
        ...
        entry_points={
            'smqtk_plugins': 'my_plugins = OtherPackage.other_classifier'
        }
    )

If the other module had multiple sub-modules in which SMQTK plugins were
defined the ``entry_points['smqtk_plugins']`` entry may instead be a list:

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
:mod:`smqtk.utils.configuration`
""""""""""""""""""""""""""""""""
.. automodule:: smqtk.utils.configuration
   :members:

:mod:`smqtk.utils.plugin`
"""""""""""""""""""""""""
.. automodule:: smqtk.utils.plugin
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
