SMQTK DIY-AI Pending Release Notes
==================================

Major updates in this release:
- Deprecation of python 2.7 support. SMQTK now requires python 3.6 or greater.


Updates / New Features
----------------------

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

General

* Fixed various deprecation warnings due to use of ABCs directly from
  ``collections``, deprecated assert methods, and invalid escape
  sequences

* Fixed broken link in top-level ``README.md`` and added a quick-start to
  building local documentation.

Docker

* Fixed IQR Playground build.

Representations

* DescriptorSet

  * Fixed missing return statement in ``DescriptorSet.get_many_vectors``.

Misc.

* Fixed various small formatting issues raised by new use of flake8.

* Fixed various small issues raised by new use of mypy.

Utils

* Replaced use of deprecated function ``logging.Logger.warn``.

* Removed some uses of ``six`` in connection with the Python 2.7
  deprecation.

* Updated configuration constructor inspection to use ``signature`` and handle
  keyword-only parameters.
