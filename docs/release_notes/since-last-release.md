Changes Since Last Release
==========================


Updates / New Features since v0.2.1
-----------------------------------

Classifiers

  * Added generic Classifier algorithm interface.

Classification Elements

  * Added classification result encapsulation interface.

  * Added in-memory implementation

  * Added ClassificationElementFactory implementation.

Data Elements

  * Added DataFileElement implementation the optional use of the tika module
    for file content type extraction. Falls back to previous method when tika
    module not found or fails.

Descriptor Generators

  * Removed ``PARALLEL`` class variable (parameterized in pertinent
    implementation constructors).

  * Added ``CaffeDescriptorGenerator`` implementation, which is more
    generalized and model agnostic, using the Caffe python interface.

Tools / Scripts

  * Added descriptor compute script that reads from a file-list text file
    specifying input data file paths, and asynchronously computes descriptors.
    Uses JSON configuration file for algorithm and element backend
    specification.

Web / Services

  * Added ``NearestNeighborServiceServer``, which provides
    web-service that returns the nearest `N` neighbors to the given
    descriptor element.

Fixes since v0.2.1
------------------

Descriptor Elements

  * Fix threading joining in ``elements_to_matrix`` (when using
    non-multiprocessing mode).

  * Fixed configuration use in ``DescriptorElementFactory.from_config``.