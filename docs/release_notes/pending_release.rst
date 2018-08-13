SMQTK Pending Release Notes
===========================


Updates / New Features since v0.8.1
-----------------------------------

General

- Added support for Python 3.
- Made some optimizations to the Postgres database access.

Travis CI

- Removed use of Miniconda installation since it wasn't being utilized in
  special way.

Fixes since v0.8.1
------------------

Tests

- Fixed ambiguous ordering check in libsvm-hik implementation of
  RelevancyIndex algorithm.
