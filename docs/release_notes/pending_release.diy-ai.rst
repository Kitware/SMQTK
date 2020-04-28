SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

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
