SMQTK MIPOL Pending Release Notes
=================================


Updates / New Features
----------------------

Algorithms

- Updated SupervisedClassifier abstract interface to use the template pattern
  with the train method. Now, implementing classes need to define
  ``_train``. The ``train`` method is not abstract anymore and calls the
  ``_train`` method after the input data consolidation.

Docker

- Versioning changes to, by default, encode date built instead of arbitrary
  separate versioning.

Fixes
-----
