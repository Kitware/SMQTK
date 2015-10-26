SMQTK Architecture Overview
===========================

SMQTK is mainly comprised of 4 high level components, with additional sub-modules for tests, utilities and other control structures.

.. toctree::
  :maxdepth: 3

  dataabstraction
  algorithms
  webservices
  utilities

Plugin Architecture
-------------------

Each of these main components are housed within distinct sub-modules under ``smqtk`` and adhere to a plugin pattern for the dynamic discovery of implementations.

In SMQTK, data structures and algorithms are first defined by an abstract interface class that lays out what that services the data structure, or methods that the algorithm, should provide.
This allows users to treat instances of structures and algorithms in a generic way, based on their defined high level functionality, without needing to knowing what specific implementation is running underneath.
It lies, of course, to the implementations of these interfaces to provide the concrete functionality.

When creating a new data structure or algorithm interface, the pattern is that each interface is defined inside its own sub-module in the ``__init__.py`` file.
This file also defines a function ``get_..._impls()`` (replacing the ``...`` with the name of the interface) that returns a mapping of implementation class names to the implementation class type, by calling the general helper method [``smqtk.utils.plugin.get_plugins``](/python/smqtk/utils/plugin.py#L31).
This helper method looks for modules defined parallel to the ``__init__.py`` file and extracts classes that extend from the specified interface class as specified by a specified helper variable or by matching the file's name to a contained class name.
See the doc-string of [``smqtk.utils.plugin.get_plugins``](/python/smqtk/utils/plugin.py#L31) for more information on how plugin modules are discovered.

Adding a new Interface and Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For example, lets say we're creating a new data structure interface called ``FooBar``.
We would create a directory and ``__init__.py`` file (python module) to house the interface as follows::

    python/
    └── smqtk/
        └── representation/
            └── foo_bar/          # new
                └── __init__.py   # new


The ``__init__.py`` file might look something like the following, defining a new abstract class (sets or descends from something that sets ``__metaclass__ = abc.ABCMeta``):

.. code-block:: python

    import abc

    from smqtk.utils.configurable_interface import Configurable

    class FooBar (Configurable):
        """
        Some documentation on what this does.
        """
        # Interface methods and/or abstract functionality here

    def get_foo_bar_impls(reload_modules=False):
        import os.path as osp
        from smqtk.utils.plugin import get_plugins
        this_dir = osp.abspath(osp.dirname(__file__))
        helper_var = 'FOO_BAR_CLASS'
        return get_plugins(__name__, this_dir, helper_var, CodeIndex, None,
                           reload_modules)


In order to allow for implementations of an interface to to be configurable, as well allowing for implementations used to be determined at configuration time, new interfaces should descend from the ``Configurable`` interface class.
This interface class sets ``__metaclass__ = abc.ABCMeta``, thus it is not set in the example above.

When adding a an implementation class, if it is sufficient to be contained in a single file, a new module can be added like::

    python/
    └── smqtk/
        └── representation/
            └── foo_bar/
                ├── __init__.py
                └── some_impl.py  # new

Where ``some_impl.py`` might look like:

.. code-block:: python

    from smqtk.representation.foo_bar import FooBar

    class SomeImpl (FooBar):
        """
        Some documentation
        """
        # Implementation of some stuff

    FOO_BAR_CLASS = SomeImpl

It is important to note the ``FOO_BAR_CLASS = SomeImpl`` line (where ``FOO_BAR_CLASS`` is what is specified to the ``helper_var`` in the ``get_foo_bar_impls`` function).
This is important to include because this allows the plugin helper to know what class to import (or multiple classes if its set to an iterable).

Implementation classes can also live inside of a nested sub-module.
This is useful when an implementation class requires extensive, specific support utilities (for example, see the ``DescriptorGenerator`` implementation [``ColorDescriptor``](/python/smqtk/algorithms/descriptor_generator/colordescriptor)).::

    python/
    └── smqtk/
        └── representation/
            └── foo_bar/
                ├── __init__.py
                ├── some_impl.py
                └── other_impl/      # new
                    └── __init__.py  # new

Where the ``__init__.py`` file should at least define the helper variable reference to implementation classes that should be exported.


Reload Use Warning
""""""""""""""""""

While the [``smqtk.utils.plugin.get_plugins``](/python/smqtk/utils/plugin.py) function allows for reloading discovered modules for potentially new content, this is not recommended under normal conditions.
When reloading a plugin module after ``pickle`` serializing an instance of an implementation, deserialization causes an error because the original class type that was pickled is no longer valid as the reloaded model overwrote the previous plugin class type.
