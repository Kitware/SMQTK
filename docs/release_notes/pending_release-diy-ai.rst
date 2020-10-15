SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

CI

* Deprecate the use of Drone, changing to the use of GitLab runner
  configuration for use in that context.

* Added flake8 and mypy checks to the gitlab task list for format and type
  checking, respectively.

Misc.

* Updated requirements file layout. Now there is a single ``requirements.txt``
  in the root directory, with other requirements files within the
  ``requirements/`` directory.

* Deprecate the use of the ``setup.cfg`` file.

Utils

* bits

  * Removed unused JIT-decorated functions, which also removed unused optional
    dependency on numba.

* Configuration

  * In configuration dictionaries we now use the fully python module path
    instead of just the leaf class name. This change is an effort to prevent
    naming conflicts between plugins that happen to share the same class name
    but are located in different module paths.

* Plugin

  * Added an optional discovery method that uses the `__subclasses__` built-in
    method on types to introspect sub-class types defined anywhere in the
    current interpreter scope. This is to satisfy the use-case where a user has
    defined an implementation type locally when the other discovery methods
    would otherwise miss it.


Fixes
-----

Algorithms

* NearestNeighborIndex

  * FAISS

    * Fix attribute reference missing issue when the installed FAISS
      package does not support GPU.

Misc.

* Fixed various small formatting issues raised by new use of flake8.

* Fixed various small issues raised by new use of mypy.
