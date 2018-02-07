SMQTK MIPOL Pending Release Notes
=================================


Updates / New Features
----------------------

Algorithms

- Updated SupervisedClassifier abstract interface to use the template pattern
  with the train method. Now, implementing classes need to define
  ``_train``. The ``train`` method is not abstract anymore and calls the
  ``_train`` method after the input data consolidation.

- Update API of classifier to support use of generic extra training parameters.

Docker

- Versioning changes to, by default, encode date built instead of arbitrary
  separate versioning.

Representation

- Descriptor Element

  - Added a return of self to vector setting method.

Testing

- Added more tests for Flask-based web services.


Fixes
-----

Misc

- Various typo fixes in in-code documentation.
