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

Classifiers

* LibSVM

  * Added `n_jobs` optional constructor argument to control new multiprocessing
    parallel prediction within `_classify_arrays` implementation.

Docker

* Migrated build logic for caffe and iqr-playground into
  `docker-compose.build.yml` configuration file.

Utils

* Expand ``parallel_map`` function documentation.

* Added daemon flag to ``parallel_map``, defaulted to True, that flags
  threads/processes created as daemonic in behavior.

* Added `smqtk.utils.file.safe_file_context` to provide a contextmanager
  complement to `safe_file_write` in the same module.

Web

* Classifier Service

  * Added optional configuration of a `DescriptorSet`.

  * Added endpoint to classify descriptors within the configured
    `DescriptorSet` given a list of descriptor UIDs.


Fixes
-----

General

* Fixed various deprecation warnings due to use of ABCs directly from
  ``collections``, deprecated assert methods, and invalid escape
  sequences

* Fixed broken link in top-level `README.md` and added a quick-start to
  building local documentation.

Docker

* Fixed IQR Playground build.

Representations

* DescriptorSet

  * Fixed missing return statement in `DescriptorSet.get_many_vectors`.

Utils

* Replaced use of deprecated function ``logging.Logger.warn``.

* Removed some uses of ``six`` in connection with the Python 2.7
  deprecation.

* Updated configuration constructor inspection to use `signature` and handle
  keyword-only parameters.
