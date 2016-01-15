Changes Since Last Release
==========================


Updates / New Features since v0.2.2
-----------------------------------

Custom LibSVM

  * Fix compiler error on Windows with Visual Studio < 2013.  Log2 doesn't exist
    until that VS version.  Added stand-in.

DescriptorIndex

  * Added initial Solr backend implementation.

Tests

  * Added tests for DescriptorIndex abstract and in-memory implementation.

Tools / Scripts

  * Added optional global default config generation to ``summarizePlugins.py``

Utilities

  * Added ``cosine_distance`` function (inverse of ``cosine_similarity``)

  * Updated ``compute_distance_kernel`` to be able to take ``numba.jit``
    compiled functions

Web / Services

  * Added query sub-slice return option to NearestNeighborServiceServer web-app.


Fixes since v0.2.2
------------------

DescriptorElement

  * Fixed mutibility of stored descriptors in DescriptorMemoryElement
    implementation.

Tools / Scripts

  * Added ``Classifier`` interface plugin summarization to
    ``summarizePlugins.py``.

Web / Services

  * Fixed issue with IQR alerts not showing whitespace correctly.

  * Fixed issue with IQR reset not resetting everything, which caused the
    application to become unusable.
