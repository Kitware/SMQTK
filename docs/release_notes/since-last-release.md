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

  * Added example of setting up a NearestNeighborServiceServer with live-reload
    enabled and how to add/process incremental ingests.

Tools / Scripts

  * Added CLI script for hash code generation and output to file. This script
    is primarily for support of LSHNearestNeighborIndex live-reload
    functionality.

  * Added script for asynchronously computing classifications on descriptors
    in an index via a list of descriptor UUIDs.

  * Added script for cross validating a classifier configuration for some
    truthed descriptors within an index.

Utilities

  * Added helper wrapper for generalized asynchronous function mapping to an
    input stream.

  * Added helper function for loop progress reporting and timing.

  * Added helper function for JSON configuration loading.

  * Added helper for utilities, encapsulating standard argument parser and
    configuration loading/generation steps.

  * Renamed "merge_config" to "merge_dict" and moved it to the smqtk.utils
    module level.


Fixes since v0.3.0
------------------

ClassificationElement

  * Fixed memory implementation serialization bug.
