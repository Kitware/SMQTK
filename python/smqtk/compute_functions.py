"""
Collection of higher level functions to perform operational tasks.

Some day, this module could have a companion module containing the CLI logic
for these functions instead of scripts in ``<source>/bin/scripts``.

"""
import collections
import logging

import numpy
import six

from smqtk.utils import (
    bin_utils,
    bit_utils,
    parallel,
)


def compute_many_descriptors(data_elements, descr_generator, descr_factory,
                             descr_index, batch_size=None, overwrite=False,
                             procs=None, **kwds):
    """
    Compute descriptors for each data element, yielding
    (DataElement, DescriptorElement) tuple pairs in the order that they were
    input.

    *Note:* **This function currently only operated over images due to the
    specific data validity check/filter performed.*

    :param data_elements: Iterable of DataElement instances of files to
        work on.
    :type data_elements: collections.Iterable[smqtk.representation.DataElement]

    :param descr_generator: DescriptorGenerator implementation instance
        to use to generate descriptor vectors.
    :type descr_generator: smqtk.algorithms.DescriptorGenerator

    :param descr_factory: DescriptorElement factory to use when producing
        descriptor vectors.
    :type descr_factory: smqtk.representation.DescriptorElementFactory

    :param descr_index: DescriptorIndex instance to add generated descriptors
        to. When given a non-zero batch size, we add descriptors to the given
        index in batches of that size. When a batch size is not given, we add
        all generated descriptors to the index after they have been generated.
    :type descr_index: smqtk.representation.DescriptorIndex

    :param batch_size: Optional number of elements to asynchronously compute
        at a time. This is useful when it is desired for this function to yield
        results before all descriptors have been computed, yet still take
        advantage of any batch asynchronous computation optimizations a
        particular DescriptorGenerator implementation may have. If this is 0 or
        None (false-evaluating), this function blocks until all descriptors have
        been generated.
    :type batch_size: None | int | long

    :param overwrite: If descriptors from a particular generator already exist
        for particular data, re-compute the descriptor for that data and set
        into the generated DescriptorElement.
    :type overwrite: bool

    :param procs: Tell the DescriptorGenerator to use a specific number of
        threads/cores.
    :type procs: None | int

    :param kwds: Remaining keyword-arguments that are to be passed into the
        ``compute_descriptor_async`` function on the descriptor generator.
    :type kwds: dict

    :return: Generator that yields (DataElement, DescriptorElement) for each data
        element given, in the order they were provided.
    :rtype: __generator[(smqtk.representation.DataElement, smqtk.representation.DescriptorElement)]

    """
    log = logging.getLogger(__name__)

    # Capture of generated elements in order of generation
    #: :type: deque[smqtk.representation.DataElement]
    de_deque = collections.deque()

    # Counts for logging
    total = 0
    unique = 0

    def iter_capture_elements():
        for de in data_elements:
            de_deque.append(de)
            yield de

    if batch_size:
        log.debug("Computing in batches of size %d", batch_size)

        batch_i = 0

        for de in iter_capture_elements():
            # elements captured ``de_deque`` in iter_capture_elements

            if len(de_deque) == batch_size:
                batch_i += 1
                log.debug("Computing batch %d", batch_i)

                total += len(de_deque)
                m = descr_generator.compute_descriptor_async(
                    de_deque, descr_factory, overwrite, procs, **kwds
                )
                unique += len(m)
                log.debug("-- Processed %d so far (%d total data elements "
                          "input)", unique, total)

                log.debug("-- adding to index")
                descr_index.add_many_descriptors(six.itervalues(m))

                log.debug("-- yielding generated descriptor elements")
                for e in de_deque:
                    # noinspection PyProtectedMember
                    yield e, m[e.uuid()]

                de_deque.clear()

        if len(de_deque):
            log.debug("Computing final batch of size %d",
                      len(de_deque))

            total += len(de_deque)
            m = descr_generator.compute_descriptor_async(
                de_deque, descr_factory, overwrite, procs, **kwds
            )
            unique += len(m)
            log.debug("-- Processed %d so far (%d total data elements "
                      "input)", unique, total)

            log.debug("-- adding to index")
            descr_index.add_many_descriptors(six.itervalues(m))

            log.debug("-- yielding generated descriptor elements")
            for de in de_deque:
                # noinspection PyProtectedMember
                yield de, m[de.uuid()]

    else:
        log.debug("Using single async call")

        # Just do everything in one call
        log.debug("Computing descriptors")
        m = descr_generator.compute_descriptor_async(
            iter_capture_elements(), descr_factory,
            overwrite, procs, **kwds
        )

        log.debug("Adding to index")
        descr_index.add_many_descriptors(six.itervalues(m))

        log.debug("yielding generated elements")
        for de in de_deque:
            # noinspection PyProtectedMember
            yield de, m[de.uuid()]


def compute_hash_codes(uuids, index, functor, report_interval=1.0, use_mp=False,
                       ordered=False):
    """
    Given an iterable of DescriptorElement UUIDs, asynchronously access them
    from the given ``index``, asynchronously compute hash codes via ``functor``
    and convert to an integer, yielding (UUID, hash-int) pairs.

    :param uuids: Sequence of UUIDs to process
    :type uuids: collections.Iterable[collections.Hashable]

    :param index: Descriptor index to pull from.
    :type index: smqtk.representation.descriptor_index.DescriptorIndex

    :param functor: LSH hash code functor instance
    :type functor: smqtk.algorithms.LshFunctor

    :param report_interval: Frequency in seconds at which we report speed and
        completion progress via logging. Reporting is disabled when logging
        is not in debug and this value is greater than 0.
    :type report_interval: float

    :param use_mp: If multiprocessing should be used for parallel
        computation vs. threading. Reminder: This will copy currently loaded
        objects onto worker processes (e.g. the given index), which could lead
        to dangerously high RAM consumption.
    :type use_mp: bool

    :param ordered: If the element-hash value pairs yielded are in the same
        order as element UUID values input. This function should be slightly
        faster when ordering is not required.
    :type ordered: bool

    :return: Generator instance yielding (DescriptorElement, int) value pairs.

    """
    # TODO: parallel map fetch elements from index?
    #       -> separately from compute

    def get_hash(u):
        v = index.get_descriptor(u).vector()
        return u, bit_utils.bit_vector_to_int_large(functor.get_hash(v))

    # Setup log and reporting function
    log = logging.getLogger(__name__)

    if log.getEffectiveLevel() > logging.DEBUG or report_interval <= 0:
        log_func = lambda m: None
        log.debug("Not logging progress")
    else:
        log.debug("Logging progress at %f second intervals", report_interval)
        log_func = log.debug

    log.debug("Starting computation")
    reporter = bin_utils.ProgressReporter(log_func, report_interval)
    reporter.start()
    for uuid, hash_int in parallel.parallel_map(get_hash, uuids,
                                                ordered=ordered,
                                                use_multiprocessing=use_mp):
        yield (uuid, hash_int)
        # Progress reporting
        reporter.increment_report()

    # Final report
    reporter.report()


def mb_kmeans_build_apply(index, mbkm, initial_fit_size):
    """
    Build the MiniBatchKMeans centroids based on the descriptors in the given
    index, then predicting descriptor clusters with the final result model.

    If the given index is empty, no fitting or clustering occurs and an empty
    dictionary is returned.

    :param index: Index of descriptors
    :type index: smqtk.representation.DescriptorIndex

    :param mbkm: Scikit-Learn MiniBatchKMeans instead to train and then use for
        prediction
    :type mbkm: sklearn.cluster.MiniBatchKMeans

    :param initial_fit_size: Number of descriptors to run an initial fit with.
        This brings the advantage of choosing a best initialization point from
        multiple.
    :type initial_fit_size: int

    :return: Dictionary of the cluster label (integer) to the set of descriptor
        UUIDs belonging to that cluster.
    :rtype: dict[int, set[collections.Hashable]]

    """
    log = logging.getLogger(__name__)

    ifit_completed = False
    k_deque = collections.deque()
    d_fitted = 0

    log.info("Getting index keys (shuffled)")
    index_keys = sorted(index.iterkeys())
    numpy.random.seed(mbkm.random_state)
    numpy.random.shuffle(index_keys)

    def parallel_iter_vectors(descriptors):
        """ Get the vectors for the descriptors given.
        Not caring about order returned.
        """
        return parallel.parallel_map(lambda d: d.vector(), descriptors,
                                     use_multiprocessing=False)

    def get_vectors(k_iter):
        """ Get numpy array of descriptor vectors (2D array returned) """
        return numpy.array(list(
            parallel_iter_vectors(index.get_many_descriptors(k_iter))
        ))

    log.info("Collecting iteratively fitting model")
    rps = [0] * 7
    for i, k in enumerate(index_keys):
        k_deque.append(k)
        bin_utils.report_progress(log.debug, rps, 1.)

        if initial_fit_size and not ifit_completed:
            if len(k_deque) == initial_fit_size:
                log.info("Initial fit using %d descriptors", len(k_deque))
                log.info("- collecting vectors")
                vectors = get_vectors(k_deque)
                log.info("- fitting model")
                mbkm.fit(vectors)
                log.info("- cleaning")
                d_fitted += len(vectors)
                k_deque.clear()
                ifit_completed = True
        elif len(k_deque) == mbkm.batch_size:
            log.info("Partial fit with batch size %d", len(k_deque))
            log.info("- collecting vectors")
            vectors = get_vectors(k_deque)
            log.info("- fitting model")
            mbkm.partial_fit(vectors)
            log.info("- cleaning")
            d_fitted += len(k_deque)
            k_deque.clear()

    # Final fit with any remaining descriptors
    if k_deque:
       log.info("Final partial fit of size %d", len(k_deque))
       log.info('- collecting vectors')
       vectors = get_vectors(k_deque)
       log.info('- fitting model')
       mbkm.partial_fit(vectors)
       log.info('- cleaning')
       d_fitted += len(k_deque)
       k_deque.clear()

    log.info("Computing descriptor classes with final KMeans model")
    mbkm.verbose = False
    d_classes = collections.defaultdict(set)
    d_uv_iter = parallel.parallel_map(lambda d: (d.uuid(), d.vector()),
                                      index,
                                      use_multiprocessing=False,
                                      name="uv-collector")
    # TODO: Batch predict call inputs to something larger than one at a time.
    d_uc_iter = parallel.parallel_map(
        lambda (u, v): (u, mbkm.predict(v[numpy.newaxis, :])[0]),
        d_uv_iter,
        use_multiprocessing=False,
        name="uc-collector")
    rps = [0] * 7
    for uuid, c in d_uc_iter:
        d_classes[c].add(uuid)
        bin_utils.report_progress(log.debug, rps, 1.)
    rps[1] -= 1
    bin_utils.report_progress(log.debug, rps, 0)

    return d_classes
