Changes Since Last Release
==========================


Updates / New Features since v0.2.2
-----------------------------------

CodeIndex

  * Deprecated/Removed because of duplication with ``DescriptorIndex``,
    ``HashIndex`` and LSH algorithm.

Custom LibSVM

  * Fix compiler error on Windows with Visual Studio < 2013.  Log2 doesn't exist
    until that VS version.  Added stand-in.

DescriptorIndex

  * Added initial Solr backend implementation.

HashIndex

  * Added new ``HashIndex`` algorithm interface for efficient neighbor
    indexing of hash codes (bit vectors).

  * Added linear (brute force) implementation.

  * Added ball-tree implementation (uses ``sklearn.neighbors.BallTree``)

LshFunctor

  * Added new interface for LSH hash code generation functor.

  * Added ITQ functor (replaces old ``ITQNearestNeighborsIndex``
    functionality).

NearestNeighborIndex

  * Added generalized LSH implementation: ``LSHNearestNeighborIndex``,
    which uses a combination of ``LshFunctor`` and ``HashIndex`` for
    modular assembly of functionality.

  * Removed deprecated ``ITQNearestNeighborsIndex`` implementation
    (reproducible using the new ``LSHNearestNeighborIndex`` with
    ``ItqFunctor`` and ``LinearHashIndex``).

Tests

  * Added tests for DescriptorIndex abstract and in-memory implementation.

  * Removed tests for deprecated ``CodeIndex`` and ``ITQNearestNeighborsIndex``

  * Added tests for ``LSHNearestNeighborIndex`` + high level tests using ITQ
    functor with linear and ball-tree hash indexes.

Tools / Scripts

  * Added optional global default config generation to ``summarizePlugins.py``

  * Updated ``summarizePlugins.py``, removing ``CodeIndex`` and adding
    ``LshFunctor`` and ``HashIndex`` interfaces.

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

Utilities

  * Fixed but with ``smqtk.utils.bit_utils.int_to_bit_vector[_large]`` when
    give a 0-value integer.

Web / Services

  * Fixed issue with IQR alerts not showing whitespace correctly.

  * Fixed issue with IQR reset not resetting everything, which caused the
    application to become unusable.
