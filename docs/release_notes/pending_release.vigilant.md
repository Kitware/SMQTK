SMQTK VIGILANT Pending Release Notes
====================================


Updates / New Features
----------------------

Misc.

* Resolve python static analysis warnings and errors.

Utilities

* Renamed ``smqtk.utils.configurable`` to ``smqtk.utils.configuration``.
  Ramifications fixed throughout the codebase.

* Moved some helper functions from ``smqtk.utils.plugin``to
  ``smqtk.utils.configuration`` as those functions more specifically had to do
  with configuration dictionary construction and manipulation. Ramifications
  fixed  throughout the codebase.

* Updated ``smqtk.utils.plugin.get_plugins`` signature and return. Now more
  simply takes the interface class (previously referred to as the base-class)
  instead of the original first two positional, string arguments as they could
  be easily introspected from the interface class object. Ramifications fixed
  throughout the codebase.

* Added ``ContentTypeValidator`` interface for algorithms that operate on raw
  ``DataElement`` instances, providing methods for validating reported content
  types against a sub-class defined set of "valid" types. Applied to
  ``DescriptorGenerator`` interface.
  
* Replace usage of ``smqtk.utils.bin_utils.report_progress`` with the 
  ``ProgressReporter`` class throughout package.

Fixes
-----

Tests

* Moved ``--cov`` options from pytest.ini file into the runner script.  This
  fixes debugger breakpointing in some IDEs.

Utils

* Fix ``ZeroDivisionError`` in ``smqtk.utils.bin_utils.report_progress``. Also
  added deprecation warning to this function.

Algorithms

* Nearest Neighbors

  * FAISS

    * Fix issue with storing and retrieving index IDs as numpy types by casting
      to python native integers due to an incompatibility with some
      KeyValueStore implementations (specificially an issue with the PostgreSQL
      implementation).
