SMQTK VIGILANT Pending Release Notes
====================================


Updates / New Features
----------------------

Data Structures

* Revise `GirderDataElement` to use `girder_client` python module and added the
  the use of girder authentication token values in lieu of username/password
  for communication authorization.
  
* Add the optional use of named cursors in PostgreSQL implementation of the
  `DescriptorIndex` interface.  Assists with large selects such that the server
  only sends batches of results at a time instead of the whole result pool.
  
* Added PostgreSQL implementation of the KeyValueStore interface.

Girder

* Initial SMQTK Girder plugin to support image descriptor processing via
  girder-worker.
  
* Initial SMQTK Girder plugin implementing a resource and UI for SMQTK nearest
  neighbors and IQR.


Fixes
-----

Data Structures

* Added locking to PostgreSQL `DescriptorElement` table creation to fix race
  condition when multiple elements tried to create the same table at the same
  time.

* Fix unconditional import of optional `girder_client` dependency.

Dependencies

* Pinned Pillow version requirement to 4.0.0 due to a large-image conversion
  issue that appeared in 4.1.x.  This issue may have been resolved in newer
  versions of Pillow.

Scripts

* Various fixes to IQR model generation process due to changes made to
  algorithm input parameters (i.e. taking `DataElement` instances instead of
  filepaths).

* Fixes `build_iqr_models.sh` to follow symlinks when compiling input image
  file list.
  
Tests

* Fix missing abstract function override in KeyValueStore test stub.
