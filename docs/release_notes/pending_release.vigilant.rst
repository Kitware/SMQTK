SMQTK VIGILANT Pending Release Notes
====================================


Updates / New Features
----------------------

Algorithms

* Descriptor Generators

  * Change ``CaffeDescriptorGenerator`` constructor to take ``DataElement``
    instances rather than URIs.

* Nearest Neighbors

  * FAISS

    * Gracefully handle addition of duplicated descriptors to avoid making
      index unusable due to an unexpected external failure.

  * LSH Hash Functor

    * Use ``ProgressReporter`` in itq to avoid bugs from deprecated
      ``report_progress`` function

Compute Functions

* Add ``compute_transformed_descriptors`` function to ``compute_functions.py`` for
  conducting searches with augmented copies of an image

Misc.

* Resolve python static analysis warnings and errors.

Representation

* Added ``BoundingBox`` class for describing N-dimensional euclidean spatial
  regions.

* Added ``DetectionElement`` interface, and in-memory implementation, with
  associated unit tests.

* Added ``DetectionElementFactory`` class for factory construction of
  ``DetectionElement`` instances.

* Add use of ``smqtk.utils.configuration.cls_conf_from_config_dict`` and
  ``smqtk.utils.configuration.cls_conf_to_config_dict`` to appropriate
  methods in factory classes.

Utilities

* Renamed ``smqtk.utils.configurable`` to ``smqtk.utils.configuration``.
  Ramifications fixed throughout the codebase.

* Added ``cls_conf_from_config_dict`` and ``cls_conf_to_config_dict``
  intermediate helper functions to ``smqtk.utils.configuration`` for the
  ``from_config_dict`` and ``to_config_dict`` sub-problems, respectively.
  This was motivated by duplicated functionality in element factory class
  ``from_config`` and ``get_config`` methods.

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

* Removed bundled "jsmin" in favor of using pip installed package.

* Moved ``merge_dict`` out of ``smqtk/utils/__init__.py`` and into its own
  module.

* Created ``combinatorics`` utils module, moved ``ncr`` function to here.

* Renamed various utility modules that included ``_utils`` in their name to not
  include ``_utils`` for the sake of reducing redundancy.
  
* Removed ``FileModificationMonitor`` utility class due to having no current
  use anywhere as well as its tests non-deterministically failing (issues 
  with timing and probably lack of sufficient use of mock, time to fix not 
  worth its lack of use).  The ``watchdog`` python package should be used 
  instead.

Fixes
-----

Algorithms

* Nearest Neighbors

  * FAISS

    * Fix issue with storing and retrieving index IDs as numpy types by casting
      to python native integers due to an incompatibility with some
      KeyValueStore implementations (specificially an issue with the PostgreSQL
      implementation).

Tests

* Moved ``--cov`` options from pytest.ini file into the runner script.  This
  fixes debugger breakpointing in some IDEs.

Utils

* Fix ``ZeroDivisionError`` in ``smqtk.utils.bin_utils.report_progress``. Also
  added deprecation warning to this function.
