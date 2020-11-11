Algorithm Interfaces
--------------------

.. autoclass:: smqtk.algorithms.SmqtkAlgorithm
   :members:

Here we list and briefly describe the high level algorithm interfaces which SMQTK provides.
There is at least one implementation available for each interface.
Some implementations will require additional dependencies that cannot be packaged with SMQTK.


Classifier
++++++++++
This interface represents algorithms that classify ``DescriptorElement`` instances into discrete labels or label confidences.

.. autoclass:: smqtk.algorithms.classifier.Classifier
   :members:
   :private-members:


DescriptorGenerator
+++++++++++++++++++
This interface represents algorithms that generate whole-content descriptor
vectors for one or more given input :class:`.DataElement`
instances.
The input :class:`.DataElement` instances must be of a
content type that the :class:`.DescriptorGenerator` supports, referenced
against the :meth:`~smqtk.utils.ContentTypeValidator.valid_content_types`
method (required by the :class:`~smqtk.utils.ContentTypeValidator` mixin
class).

The :meth:`.DescriptorGenerator.generate_elements` method also requires a
:class:`.DescriptorElementFactory` instance to tell the algorithm how to
generate the :class:`.DescriptorElement` instances it should return.
The returned :class:`.DescriptorElement` instances will have a type equal to
the name of the :class:`.DescriptorGenerator` class that generated it, and a
UUID that is the same as the input :class:`.DataElement` instance.

If a :class:`.DescriptorElement` implementation that supports persistent
storage is generated, and there is already a descriptor associated with the
given type name and UUID values, the descriptor is returned without
re-computation.

If the ``overwrite`` parameter is ``True``, the :class:`.DescriptorGenerator`
instance will re-compute a descriptor for the input :class:`.DataElement`,
setting it to the generated :class:`.DescriptorElement`.
This will overwrite descriptor data in persistent storage if the
:class:`.DescriptorElement` type used supports it.

.. autoclass:: smqtk.algorithms.descriptor_generator.DescriptorGenerator
   :members:


ImageReader
+++++++++++

.. autoclass:: smqtk.algorithms.image_io.ImageReader
   :members:

.. autoclass:: smqtk.algorithms.image_io.pil_io.PilImageReader
   :members:


HashIndex
+++++++++

This interface describes specialized ``NearestNeighborsIndex`` implementations designed to index hash codes (bit vectors) via the hamming distance function.
Implementations of this interface are primarily used with the ``LSHNearestNeighborIndex`` implementation.

Unlike the ``NearestNeighborsIndex`` interface from which this interface descends, ``HashIndex`` instances are build with an iterable of ``numpy.ndarray`` and ``nn`` returns a ``numpy.ndarray``.

.. autoclass:: smqtk.algorithms.nn_index.hash_index.HashIndex
   :members:


LshFunctor
++++++++++
Implementations of this interface define the generation of a locality-sensitive hash code for a given :class:`DescriptorElement`.
These are used in :class:`LSHNearestNeighborIndex` instances.

.. autoclass:: smqtk.algorithms.nn_index.lsh.functors.LshFunctor
   :members:


NearestNeighborsIndex
+++++++++++++++++++++

This interface defines a method to build an index from a set of ``DescriptorElement`` instances (``NearestNeighborsIndex.build_index``) and a nearest-neighbors query function for getting a number of near neighbors to e query ``DescriptorElement`` (``NearestNeighborsIndex.nn``).

Building an index requires that some non-zero number of ``DescriptorElement`` instances be passed into the ``build_index`` method.
Subsequent calls to this method should rebuild the index model, not add to it.
If an implementation supports persistant storage of the index, it should overwrite the configured index.

The ``nn`` method uses a single ``DescriptorElement`` to query the current index for a specified number of nearest neighbors.
Thus, the ``NearestNeighborsIndex`` instance must have a non-empty index loaded for this method to function.
If the provided query ``DescriptorElement`` does not have a set vector, this method will also fail with an exception.

This interface additionally requires that implementations define a ``count`` method, which returns the number of distinct ``DescriptorElement`` instances are in the index.

.. autoclass:: smqtk.algorithms.nn_index.NearestNeighborsIndex
   :members:


ObjectDetector
++++++++++++++
This interface defines a method to generate object detections
(:class:`~smqtk.representation.DetectionElement`) over a given
:class:`~smqtk.representation.DataElement`.

.. autoclass:: smqtk.algorithms.object_detection.ObjectDetector
   :members:


RankRelevancy
+++++++++++++

This interface defines one method: ``rank``.  The ``rank`` method
takes examples of relevant and not-relevant example descriptor vectors
as :class:`numpy.ndarray` sequences and uses them to compute relevancy
scores (on a ``[0, 1]`` scale) on a provided pool of other descriptor
vectors.

.. autoclass:: smqtk.algorithms.rank_relevancy.RankRelevancy
   :members:


RankRelevancyWithFeedback
+++++++++++++++++++++++++

This interface defines one method: ``rank_with_feedback``.  Like
:meth:`RankRelevancy.rank`, ``rank_with_feedback`` takes examples of
relevant and not-relevant example descriptor vectors as
:class:`numpy.ndarray` sequences and uses them to compute relevancy
scores (on a ``[0, 1]`` scale) on a provided pool of other descriptor
vectors.  However, it also expects a sequence of corresponding UIDs
for the pool vectors and additionally returns a sequence of UIDs,
possibly not all from the pool, on which feedback would be most
useful.

.. autoclass:: smqtk.algorithms.rank_relevancy.RankRelevancyWithFeedback
   :members:
   :private-members:


RelevancyIndex
++++++++++++++

This interface defines two methods: ``build_index`` and ``rank``.
The ``build_index`` method is, like a ``NearestNeighborsIndex``, used to build an index of ``DescriptorElement`` instances.
The ``rank`` method takes examples of relevant and not-relevant ``DescriptorElement`` examples with which the algorithm uses to rank (think sort) the indexed ``DescriptorElement`` instances by relevancy (on a ``[0, 1]`` scale).

.. autoclass:: smqtk.algorithms.relevancy_index.RelevancyIndex
   :members:
