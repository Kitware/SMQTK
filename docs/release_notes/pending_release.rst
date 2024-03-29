SMQTK Pending Release Notes
===========================

Major updates in this release:
- Deprecation of python 2.7 support. SMQTK now requires python 3.6 or greater.


Updates / New Features
----------------------

Algorithms

* Added new algorithms ``RankRelevancy`` and ``RankRelevancyWithFeedback``.
  ``RankRelevancy`` is an overhaul of the existing ``RelevancyIndex`` algorithm
  and will eventually replace it.  ``RankRelevancyWithFeedback`` is a closely
  related algorithm that additionally provides feedback requests.

  * An implementation ``RankRelevancyWithSupervisedClassifier`` is provided for
    ``RankRelevancy``, porting the existing
    ``SupervisedClassifierRelevancyIndex``.

  * An implementation ``RankRelevancyWithMarginSampledFeedback`` is provided
    for ``RankRelevancyWithFeedback``, supporting wrapping a ``RankRelevancy``
    instance for margin sampling.

* Added test checking that a pending release notes files is updated on a merge
  request, otherwise it fails (gitlab). The intent of this test is to remind
  contributors that they ought to be adding change notes.

CI

* Updated travis and drone configurations to remove python 2.7 and add 3.8.
  Also removed testing on Ubuntu Xenial (16.04) images due to lack of specific
  motivation and favoring simplicity.

* Deprecate the use of Drone, changing to the use of GitLab runner
  configuration for use in that context.

* Added flake8 and mypy checks to the gitlab task list for format and type
  checking, respectively.

Classifiers

* LibSVM

  * Added ``n_jobs`` optional constructor argument to control new
    multiprocessing parallel prediction within ``_classify_arrays``
    implementation.

Docker

* Migrated build logic for caffe and iqr-playground into
  ``docker-compose.build.yml`` configuration file.

Documentation

* Update plugin related sphinx documentation content and examples.

IQR

* Change over the previous use of the "RelevancyIndex" to the new
  "RankRelevancyWithFeedback" algorithm interface in the IqrSession class.
  Updates also reflected in the IQR web service and respective unit tests.

Misc.

* Updated requirements file layout. Now there is a single ``requirements.txt``
  in the root directory, with other requirements files within the
  ``requirements/`` directory.

* Deprecate the use of the ``setup.cfg`` file.

Utils

* Expand ``parallel_map`` function documentation.

* Added daemon flag to ``parallel_map``, defaulted to True, that flags
  threads/processes created as daemonic in behavior.

* Added ``smqtk.utils.file.safe_file_context`` to provide a contextmanager
  complement to ``safe_file_write`` in the same module.

* bits

  * Removed unused JIT-decorated functions, which also removed unused optional
    dependency on numba.

* Configuration

  * In configuration dictionaries we now use the fully python module path
    instead of just the leaf class name. This change is an effort to prevent
    naming conflicts between plugins that happen to share the same class name
    but are located in different module paths.

* Plugin

  * Added an optional discovery method that uses the ``__subclasses__``
    built-in method on types to introspect sub-class types defined anywhere in
    the current interpreter scope. This is to satisfy the use-case where a user
    has defined an implementation type locally when the other discovery methods
    would otherwise miss it.

  * Revise the underpinnings of the utilities and
    :py:class:`smqtk.utils.plugin.Pluggable` mixin class to be more modular
    and involve less special rules.

Web

* Classifier Service

  * Added optional configuration of a ``DescriptorSet``.

  * Added endpoint to classify descriptors within the configured
    ``DescriptorSet`` given a list of descriptor UIDs.


Fixes
-----

Algorithms

* NearestNeighborIndex

  * FAISS

    * Fix attribute reference missing issue when the installed FAISS
      package does not support GPU.

CMake

* Minor fix to set the appropriate working directory when fetching the version
  value.

Docker

* Fixed IQR Playground build.

* Fixed IQR Playground runtime environment by specifically installing pinned
  versions from ``requirements/runtime.txt``.

General

* Fixed various deprecation warnings due to use of ABCs directly from
  ``collections``, deprecated assert methods, and invalid escape
  sequences

* Fixed broken link in top-level ``README.md`` and added a quick-start to
  building local documentation.

Representations

* DescriptorSet

  * Fixed missing return statement in ``DescriptorSet.get_many_vectors``.

Misc.

* Fixed various small formatting issues raised by new use of flake8.

* Fixed various small issues raised by new use of mypy.

* Fix issues revealed with updating to use of mypy version 0.790.

Utils

* Replaced use of deprecated function ``logging.Logger.warn``.

* Removed some uses of ``six`` in connection with the Python 2.7
  deprecation.

* Updated configuration constructor inspection to use ``signature`` and handle
  keyword-only parameters.
