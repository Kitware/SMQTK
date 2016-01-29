Changes Since Last Release
==========================


Updates / New Features since v0.3.0
-----------------------------------

Compute Functions

  * Refactored ``compute_many_descriptors.py`` main work function into a new
    sub-module of SMQTK in in order to allow higher level compute function to
    be accessible from the SMQTK module API.

  * Added function for asynchronously computing LSH codes for some number of
    input descriptor elements.

Utilities

  * Added helper wrapper for generalized asynchronous function mapping to an
    input stream.

  * Added helper function for loop progress reporting and timing.

  * Added helper function for JSON configuration loading.


Fixes since v0.3.0
------------------
