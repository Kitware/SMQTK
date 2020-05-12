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

Docker

* Migrated build logic for caffe and iqr-playground into
  `docker-compose.build.yml` configuration file.

Utils

* Expand ``parallel_map`` function documentation.

* Add daemon flag to ``parallel_map``, defaulted to True, that flags
  threads/processes created as daemonic in behavior.


Fixes
-----

Docker

* Fixed IQR Playground build.
