"""
Collection of higher level functions to perform operational tasks.

Some day, this module could have a companion module containing the CLI logic
for these functions instead of scripts in ``<source>/bin/scripts``.

"""
import collections
import logging
import itertools

import numpy
import six
from six.moves import zip

from smqtk.utils import (
    cli,
    bits,
    parallel,
)


def compute_many_descriptors(data_elements, descr_generator, descr_factory,
                             descr_set, batch_size=None, overwrite=False,
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

    :param descr_set: DescriptorSet instance to add generated descriptors
        to. When given a non-zero batch size, we add descriptors to the given
        set in batches of that size. When a batch size is not given, we add
        all generated descriptors to the set after they have been generated.
    :type descr_set: smqtk.representation.DescriptorSet

    :param batch_size: Optional number of elements to asynchronously compute
        at a time. This is useful when it is desired for this function to yield
        results before all descriptors have been computed, yet still take
        advantage of any batch asynchronous computation optimizations a
        particular DescriptorGenerator implementation may have. If this is 0 or
        None (false-evaluating), this function blocks until all descriptors
        have been generated.
    :type batch_size: None | int | long

    :param overwrite: If descriptors from a particular generator already exist
        for particular data, re-compute the descriptor for that data and set
        into the generated DescriptorElement.
    :type overwrite: bool

    :param procs: Deprecated parameter. Parallelism in batch computation is now
        controlled on a per implementation basis.
    :type procs: None | int

    :param kwds: Deprecated parameter. Extra keyword arguments are no longer
        passed down to the batch generation method on the descriptor generator.

    :return: Generator that yields (DataElement, DescriptorElement) for each
        data element given, in the order they were provided.
    :rtype: collections.Iterable[(smqtk.representation.DataElement,
                                  smqtk.representation.DescriptorElement)]

    """
    log = logging.getLogger(__name__)

    # Capture of generated elements in order of generation
    #: :type: deque[smqtk.representation.DataElement]
    de_deque = collections.deque()

    # Counts for logging
    total = [0]
    unique = set()

    def iter_capture_elements():
        for d in data_elements:
            de_deque.append(d)
            yield d

    # TODO: Re-write this method to more simply tee the input data elem iter
    #       and yield with paired generated descriptors::
    #           data_iter1, data_iter2 = itertools.tee(data_elements, 2)
    #           descr_iter = descr_generator.generate_elements(
    #               data_iter1, descr_factory, overwrite
    #           )
    #           return zip(data_iter2, descr_iter)

    if batch_size:
        log.debug("Computing in batches of size %d", batch_size)

        def iterate_batch_results():
            descr_list_ = list(descr_generator.generate_elements(
                de_deque, descr_factory, overwrite
            ))
            total[0] += len(de_deque)
            unique.update(d.uuid() for d in descr_list_)
            log.debug("-- Processed %d so far (%d total data elements "
                      "input)", len(unique), total[0])
            log.debug("-- adding to set")
            descr_set.add_many_descriptors(descr_list_)
            log.debug("-- yielding generated descriptor elements")
            for data_, descr_ in zip(de_deque, descr_list_):
                yield data_, descr_
            de_deque.clear()

        batch_i = 0

        for _ in iter_capture_elements():
            # elements captured ``de_deque`` in iter_capture_elements

            if len(de_deque) == batch_size:
                batch_i += 1
                log.debug("Computing batch {}".format(batch_i))
                for data_e, descr_e in iterate_batch_results():
                    yield data_e, descr_e

        if len(de_deque):
            log.debug("Computing final batch of size %d",
                      len(de_deque))
            for data_e, descr_e in iterate_batch_results():
                yield data_e, descr_e

    else:
        log.debug("Using single generate call")

        # Just do everything in one call
        log.debug("Computing descriptors")
        descr_list = list(descr_generator.generate_elements(
            iter_capture_elements(), descr_factory,
            overwrite
        ))

        log.debug("Adding to set")
        descr_set.add_many_descriptors(descr_list)

        log.debug("yielding generated elements")
        for data, descr in zip(de_deque, descr_list):
            yield data, descr


class _CountedGenerator(object):
    """
    Used to count elements of an iterable as they are accessed

    :param collections.Iterable iterable: An iterable containing elements to be
        accessed and counted.
    :param list count_list: A list to which the count of items in iterable will
        be added once the iterable has been exhausted.
    """
    def __init__(self, iterable, count_list):
        self.iterable = iterable
        self.count_list = count_list
        self.count = 0

    def __call__(self):
        for item in self.iterable:
            self.count += 1
            yield item
        self.count_list.append(self.count)


def compute_transformed_descriptors(data_elements, descr_generator,
                                    descr_factory, descr_set,
                                    transform_function, batch_size=None,
                                    overwrite=False, procs=None, **kwds):
    """
    Compute descriptors for copies of each data element generated by
    a transform function, yielding a list of tuples containing the original
    DataElement as the first element and a tuple of descriptors corresponding
    to the transformed DataElements.

    *Note:* Please see the closely-related :func:`compute_many_descriptors`
    for details on parameters and usage.

    *Note:* **This function currently only operates over images due to the
    specific data validity check/filter performed.*

    :param transform_function: Takes in a DataElement and returns an iterable
        of transformed DataElements.
    :type transform_function: collections.Callable

    :rtype: collections.Iterable[
        (smqtk.representation.DataElement,
         collections.Iterable[smqtk.representation.DescriptorElement])]
    """
    transformed_counts = []

    def transformed_elements():
        for elem in data_elements:
            yield _CountedGenerator(transform_function(elem),
                                    transformed_counts)()

    chained_elements = itertools.chain.from_iterable(
        transformed_elements())
    descriptors = compute_many_descriptors(chained_elements,
                                           descr_generator, descr_factory,
                                           descr_set, batch_size=batch_size,
                                           overwrite=overwrite, procs=procs,
                                           **kwds)
    for count, de in zip(transformed_counts, data_elements):
        yield de, itertools.islice((d[1] for d in descriptors), count)


def compute_hash_codes(uuids, descr_set, functor, report_interval=1.0,
                       use_mp=False, ordered=False):
    """
    Given an iterable of DescriptorElement UUIDs, asynchronously access them
    from the given ``set``, asynchronously compute hash codes via ``functor``
    and convert to an integer, yielding (UUID, hash-int) pairs.

    :param uuids: Sequence of UUIDs to process
    :type uuids: collections.Iterable[collections.Hashable]

    :param descr_set: Descriptor set to pull from.
    :type descr_set: smqtk.representation.descriptor_set.DescriptorSet

    :param functor: LSH hash code functor instance
    :type functor: smqtk.algorithms.LshFunctor

    :param report_interval: Frequency in seconds at which we report speed and
        completion progress via logging. Reporting is disabled when logging
        is not in debug and this value is greater than 0.
    :type report_interval: float

    :param use_mp: If multiprocessing should be used for parallel
        computation vs. threading. Reminder: This will copy currently loaded
        objects onto worker processes (e.g. the given set), which could lead
        to dangerously high RAM consumption.
    :type use_mp: bool

    :param ordered: If the element-hash value pairs yielded are in the same
        order as element UUID values input. This function should be slightly
        faster when ordering is not required.
    :type ordered: bool

    :return: Generator instance yielding (DescriptorElement, int) value pairs.

    """
    # TODO: parallel map fetch elements from set?
    #       -> separately from compute

    def get_hash(u):
        v = descr_set.get_descriptor(u).vector()
        return u, bits.bit_vector_to_int_large(functor.get_hash(v))

    # Setup log and reporting function
    log = logging.getLogger(__name__)

    if log.getEffectiveLevel() > logging.DEBUG or report_interval <= 0:
        def log_func(*_, **__):
            return
        log.debug("Not logging progress")
    else:
        log.debug("Logging progress at %f second intervals", report_interval)
        log_func = log.debug

    log.debug("Starting computation")
    reporter = cli.ProgressReporter(log_func, report_interval)
    reporter.start()
    for uuid, hash_int in parallel.parallel_map(get_hash, uuids,
                                                ordered=ordered,
                                                use_multiprocessing=use_mp):
        yield (uuid, hash_int)
        # Progress reporting
        reporter.increment_report()

    # Final report
    reporter.report()


def mb_kmeans_build_apply(descr_set, mbkm, initial_fit_size):
    """
    Build the MiniBatchKMeans centroids based on the descriptors in the given
    set, then predicting descriptor clusters with the final result model.

    If the given set is empty, no fitting or clustering occurs and an empty
    dictionary is returned.

    :param descr_set: set of descriptors
    :type descr_set: smqtk.representation.DescriptorSet

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

    log.info("Getting set keys (shuffled)")
    set_keys = sorted(six.iterkeys(descr_set))
    numpy.random.seed(mbkm.random_state)
    numpy.random.shuffle(set_keys)

    def parallel_iter_vectors(descriptors):
        """ Get the vectors for the descriptors given.
        Not caring about order returned.
        """
        return parallel.parallel_map(lambda d: d.vector(), descriptors,
                                     use_multiprocessing=False)

    def get_vectors(k_iter):
        """ Get numpy array of descriptor vectors (2D array returned) """
        return numpy.array(list(
            parallel_iter_vectors(descr_set.get_many_descriptors(k_iter))
        ))

    log.info("Collecting iteratively fitting model")
    pr = cli.ProgressReporter(log.debug, 1.0).start()
    for i, k in enumerate(set_keys):
        k_deque.append(k)
        pr.increment_report()

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
    pr.report()

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
                                      descr_set,
                                      use_multiprocessing=False,
                                      name="uv-collector")
    # TODO: Batch predict call inputs to something larger than one at a time.
    d_uc_iter = parallel.parallel_map(
        lambda u_v: (u_v[0], mbkm.predict(u_v[1][numpy.newaxis, :])[0]),
        d_uv_iter,
        use_multiprocessing=False,
        name="uc-collector")
    pr = cli.ProgressReporter(log.debug, 1.0).start()
    for uuid, c in d_uc_iter:
        d_classes[c].add(uuid)
        pr.increment_report()
    pr.report()

    return d_classes
