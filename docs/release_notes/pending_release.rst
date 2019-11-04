SMQTK Pending Release Notes
===========================
This release incorporates updates and fixes performed on the VIGILANT project
and approved for public release (case number 88ABW-2019-5287).
Some of the major updates and fixes in this include:

- Object detection algorithm interface and supporting DetectionElement
  interface and implementations.
- Revised plugin implementation accessor via the mixin class instead what used
  to be manually implemented side-car functions for every interface. Also moved
  some configuration specific functions out of the plugin utility module and
  into a configuration utility submodule, where the ``Configurable`` mixin
  class has also moved to.
- Moves unit tests out of the installed SMQTK package and into a dedicated
  sub-directory in the repository.


Updates / New Features
----------------------

Algorithms

* Added ``ImageReader`` algorithm interface

  * Added matrix reading short-cut if DataElement instance provided has a
    ``matrix`` attribute/property.

  * Added PIL (pillow) implementation with tests.

  * Added GDAL implementation with tests.

* Descriptor Generators

  * Change ``CaffeDescriptorGenerator`` constructor to take ``DataElement``
    instances rather than URIs.

* HashIndex

  * SkLearnBallTreeHashIndex

    * Fixed numpy load call to explicitly allow loading pickled components due
      to a parameter default change in numpy version 1.16.3.

* Object Detection

  * Added initial abstract interface.

  * Added "ImageMatrixObjectDetector" interface for object detectors that
    specifically operate on image data and standardizes the use of an
    "ImageReader" algorithm to provide the pixel matrix as input.

* Nearest Neighbors

  * FAISS

    * Gracefully handle addition of duplicated descriptors to avoid making
      index unusable due to an unexpected external failure.

    * Make use of new ``get_many`` method of key-value stores to improve
      runtime performance.

    * Make use of new ``get_many_vectors`` classmethod of DescriptorElement to
      improve runtime performance.

  * LSH Hash Functor

    * Use ``ProgressReporter`` in itq to avoid bugs from deprecated
      ``report_progress`` function

Compute Functions

* Add ``compute_transformed_descriptors`` function to ``compute_functions.py`` for
  conducting searches with augmented copies of an image

Misc.

* Updated numpy version in requirements.txt to current versions. Also split
  versioning between python 2 and 3 due to split availability.

* Resolve python static analysis warnings and errors.

Representation

* Added ``AxisAlignedBoundingBox`` class for describing N-dimensional euclidean spatial
  regions.

* Added ``DetectionElement`` interface, and in-memory implementation, with
  associated unit tests.

* Added ``DetectionElementFactory`` class for factory construction of
  ``DetectionElement`` instances.

* Add use of ``smqtk.utils.configuration.cls_conf_from_config_dict`` and
  ``smqtk.utils.configuration.cls_conf_to_config_dict`` to appropriate
  methods in factory classes.

* Add ``get_many`` method to ``KeyValueStore`` interface class and provide an
  optimized implementation of it for the ``PostgresKeyValueStore``
  implementation class.

* Add ``get_many_vectors`` classmethod for efficiently retrieving vectors from
  several descriptor elements at once

* Add efficient implementation of ``_get_many_vectors`` for Postgres descriptor
  elements.

* Updated ``MemoryKeyValueStore.add_many`` to use ``dict.update`` method
  instead of manually updating keys.

* Removed unnecessary method override in ``DataFileElement``.

* Added ``MatrixDataElement`` representation that stores a ``numpy.ndarray``
  instance internally, generating bytes on-the-fly when requested.

* ``DataMemoryElement`` now raises a TypeError if a non-bytes-line object is
  passed during construction or setting of bytes. Configuration mixin hooks
  have been updated to convert to and from strings for JSON-compliant
  dictionary input and output. Fixed various usages of DataMemoryElement to
  actually pass bytes.

Tests

* Moved tests out of main package tree.

* Added use of ``pytest-runner`` in ``setup.py``, removing ``run_tests.sh``
  script.  New method of running tests is ``python setup.py test``.

Utilities

* Added to ``Pluggable`` interface the ``get_impls`` method, replacing the
  separate ``get_*_impls`` functions defined for each interface type.  Removed
  previous ``get_*_impls`` functions from algorithm and representation
  interfaces, adjusting tests and utilities as appropriate.

* Renamed ``smqtk.utils.configurable`` to ``smqtk.utils.configuration``.
  Ramifications fixed throughout the codebase. Added documentation to
  doc-strings.

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

* Added entry-point extension method of plugin discovery.

* Added warning to ``smqtk.utils.file.safe_file_write`` when used on Windows
  platforms.

Fixes
-----

Algorithms

* Nearest Neighbors

  * FAISS

    * Fix issue with storing and retrieving index IDs as numpy types by casting
      to python native integers due to an incompatibility with some
      KeyValueStore implementations (specificially an issue with the PostgreSQL
      implementation).

Representation

* Fixed bug with ``ClassificationElement.max_label`` where an exception would
  be raised if there was no label with associated confidence greater than 0.

* Fix some postgres test comparisons due to not being able to ``byte`` case
  Binary instances in python 3. Instead using the ``getquoted`` conversion for
  the sake of actual/expected comparisons.

Tests

* Moved ``--cov`` options from pytest.ini file into the runner script.  This
  fixes debugger break-pointing in some IDEs (e.g. PyCharm).

* Fix various minor testing errors.

Utilities

* Fix ``ZeroDivisionError`` in ``smqtk.utils.bin_utils.report_progress``. Also
  added deprecation warning to this function.
