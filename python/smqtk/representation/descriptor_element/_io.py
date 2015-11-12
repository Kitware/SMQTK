import logging
import multiprocessing
import multiprocessing.queues
import sys
import time

import numpy

from smqtk.utils import SmqtkObject


__author__ = 'paul.tunison@kitware.com'


__all__ = [
    'elements_to_matrix',
]


def elements_to_matrix(descr_elements, mat=None, procs=None, buffer_factor=2,
                       report_interval=None):
    """
    Add to or create a numpy matrix, adding to it the vector data contained in
    a sequence of DescriptorElement instances using asynchronous processing.

    If ``mat`` is provided, its shape must equal:
        ( len(descr_elements) , descr_elements[0].size )

    :param descr_elements: Sequence of DescriptorElement objects to transform
        into a matrix. Each element should contain descriptor vectors of the
        same size.
    :type descr_elements:
        collections.Sequence[smqtk.representation.DescriptorElement]

    :param mat: Optionally pre-constructed numpy matrix of the appropriate shape
        to as loaded vectors into. If supplied this must have rows of the shape:
        (nDescriptors, nFeatures)
    :type mat: None | numpy.core.multiarray.ndarray

    :param procs: Optional specification of the number of cores to use. If None,
        we will attempt to use all available cores.
    :type procs: None | int | long

    :param buffer_factor: Multiplier against the number of processes used to
        limit the growth size of the result queue coming from worker processes.
    :type buffer_factor: float

    :param report_interval: Optional interval in seconds for debug logging to
        occur reporting about conversion speed. This should be greater than 0
        if this debug logging is desired.
    :type report_interval: None | float

    :return: Created or input matrix.
    :rtype: numpy.core.multiarray.ndarray

    """
    log = logging.getLogger(__name__)

    # Special case for in-memory storage of descriptors
    from smqtk.representation.descriptor_element.local_elements \
        import DescriptorMemoryElement

    # Create/check matrix
    if mat is None:
        shp = (len(descr_elements),
               descr_elements[0].vector().size)
        log.debug("Creating new matrix with shape: %s", shp)
        mat = numpy.ndarray(shp, float)
    else:
        assert mat.shape[0] == len(descr_elements)
        assert mat.shape[1] == descr_elements[0].vector().size

    if procs is None:
        procs = multiprocessing.cpu_count()

    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue(int(procs * buffer_factor))
    log.debug("Output queue size: %d", out_q._maxsize)

    # Workers for async extraction
    log.debug("constructing worker processes")
    workers = [_ElemVectorExtractor(i, in_q, out_q) for i in xrange(procs)]

    try:
        # Start worker processes
        log.debug("starting worker processes")
        for w in workers:
            w.daemon = True
            w.start()

        log.debug("Sending work packets")
        async_packets_sent = 0
        for r, d in enumerate(descr_elements):
            if isinstance(d, DescriptorMemoryElement):
                # No loading required, already in memory
                mat[r] = d.vector()
            else:
                in_q.put((r, d))
                async_packets_sent += 1

        # Collect work from async
        log.debug("Aggregating async results")
        f = 0
        lt = t = time.time()
        for _ in xrange(async_packets_sent):
            r, v = out_q.get()
            mat[r] = v

            f += 1
            if report_interval and time.time() - lt >= report_interval:
                log.debug("Rows per second: %f, Total: %d",
                          f / (time.time() - t), f)
                lt = time.time()

        log.debug("Closing output queue")
        out_q.close()

        # All work should be exhausted at this point
        if sys.platform == 'darwin':
            # qsize doesn't work on OSX
            # Try to get something from each queue, expecting an empty exception
            try:
                in_q.get(block=False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise AssertionError("In queue not empty")
            try:
                out_q.get(block=False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise AssertionError("Out queue not empty")
        else:
            assert in_q.qsize() == 0, "In queue not empty"
            assert out_q.qsize() == 0, "Out queue not empty"

        # Shutdown workers
        # - Workers should exit upon getting a None packet
        log.debug("Sending worker terminal signals")
        for _ in workers:
            # One for each worker
            in_q.put(None)
        log.debug("Closing input queue")
        in_q.close()

        log.debug("Done")
        return mat
    finally:
        # Forcibly terminate worker processes if still alive
        log.debug("Joining/Terminating workers")
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join()
        for q in (in_q, out_q):
            q.close()
            q.join_thread()


class _ElemVectorExtractor (SmqtkObject, multiprocessing.Process):
    """
    Helper process for extracting DescriptorElement vectors on a separate
    process. This terminates with a None packet fed to in_q. Otherwise, in_q
    values are expected to be (row, element) pairs. Tuples of the form
    (row, vector) are published to the out_q.

    Terminal value: None

    """

    def __init__(self, i, in_q, out_q):
        super(_ElemVectorExtractor, self)\
            .__init__(name='[w%d]' % i)
        self._log.debug("Making worker (%d, %s, %s)", i, in_q, out_q)
        self.i = i
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        packet = self.in_q.get()
        while packet is not None:
            row, elem = packet
            self.out_q.put((row, elem.vector()))
            packet = self.in_q.get()
