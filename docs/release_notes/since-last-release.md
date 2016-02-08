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
    
Documentation

  * Added ``CONTRIBUTING.md`` file.

Tools / Scripts

  * Added CLI script for hash code generation and output to file. This script
    is primarilly for support of LSHNearestNeighborsIndex live-reload
    functionality.

  * Added script for asynchronously computing classifications on descriptors
    in an index via a list of descriptor UUIDs.

Utilities

  * Added helper wrapper for generalized asynchronous function mapping to an
    input stream.

  * Added helper function for loop progress reporting and timing.

  * Added helper function for JSON configuration loading.

  * Added helper for utilities, encapsulating standard argument parser and
    configuration loading/generation steps.


Fixes since v0.3.0
------------------

ClassificationElement

  * Fixed memory implementation serialization bug.
