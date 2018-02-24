import logging
import multiprocessing
import multiprocessing.queues
import sys
import threading
import time

from six.moves import queue
from six import next

import numpy

from smqtk.utils import SmqtkObject


__author__ = 'paul.tunison@kitware.com'


__all__ = [
    'elements_to_matrix',
]


def elements_to_matrix(descr_elements, mat=None, procs=None, buffer_factor=2,
                       report_interval=None, use_multiprocessing=False,
                       thread_q_put_interval=0.001):
    """
    Add to or create a numpy matrix, adding to it the vector data contained in
    a sequence of DescriptorElement instances using asynchronous processing.

    If ``mat`` is provided, its shape must equal:
        ( len(descr_elements) , descr_elements[0].size )

    :param descr_elements: Sequence of DescriptorElement objects to transform
        into a matrix. Each element should contain descriptor vectors of the
        same size.
    :type descr_elements:
        collections.Sequence[smqtk.representation.DescriptorElement] |
        collections.Iterable[smqtk.representation.DescriptorElement]

    :param mat: Optionally a pre-constructed numpy matrix of the shape
        ``(nDescriptors, nFeatures)`` to load descriptor vectors into. We will
        only iterate ``nDescriptors`` into the given ``descr_elements``
        iterable. If there are more rows in the given matrix than there are
        DescriptorElements in ``descr_elements``, then not all rows in the
        given matrix will be set. Elements yielded by ``descr_elements`` must
        be of the same dimensionality as this given matrix (``nFeatures``)
        otherwise an exception will be raised (``ValueError``, by numpy).

        If this is not supplied, we create a new matrix to insert vectors into
        based on the number of input descriptor elements. This mode required
        that the input elements are in a container that defines __len__
    :type mat: None | numpy.core.multiarray.ndarray

    :param procs: Optional specification of the number of threads/cores to use.
        If None, we will attempt to use all available threads/cores.
    :type procs: None | int | long

    :param buffer_factor: Multiplier against the number of processes used to
        limit the growth size of the result queue coming from worker processes.
    :type buffer_factor: float

    :param report_interval: Optional interval in seconds for debug logging to
        occur reporting about conversion speed. This should be greater than 0
        if this debug logging is desired.
    :type report_interval: None | float

    :param use_multiprocessing: Whether or not to use discrete processes as the
        parallelization agent vs python threads.
    :type use_multiprocessing: bool

    :param thread_q_put_interval: Interval at worker threads attempt to insert
        values into the output queue after fetching vector from a
        DescriptorElement. This is for dead-lock protection due to size-limited
        output queue. This is only used if ``use_multiprocessing`` is ``False``
        and this must be >0.
    :type thread_q_put_interval: float

    :return: Created or input matrix.
    :rtype: numpy.core.multiarray.ndarray

    """
    log = logging.getLogger(__name__)

    # Create/check matrix
    if mat is None:
        sample = next(iter(descr_elements))
        sample_v = sample.vector()
        shp = (len(descr_elements),
               sample_v.size)
        log.debug("Creating new matrix with shape: %s", shp)
        mat = numpy.ndarray(shp, sample_v.dtype)

    if procs is None:
        procs = multiprocessing.cpu_count()

    # Choose parallel types
    worker_kwds = {}
    if use_multiprocessing:
        queue_t = multiprocessing.Queue
        worker_t = _ElemVectorExtractorProcess
    else:
        queue_t = queue.Queue
        worker_t = _ElemVectorExtractorThread

        assert thread_q_put_interval >= 0, \
            "Thread queue.put interval must be >= 0. (given: %f)" \
            % thread_q_put_interval
        worker_kwds['q_put_interval'] = thread_q_put_interval

    in_q = queue_t()
    out_q = queue_t(int(procs * buffer_factor))
    # Workers for async extraction
    log.debug("constructing worker processes")
    workers = [worker_t(i, in_q, out_q, **worker_kwds) for i in range(procs)]

    in_queue_t = _FeedQueueThread(descr_elements, in_q, mat, len(workers))

    try:
        # Start worker processes
        log.debug("starting worker processes")
        for w in workers:
            w.daemon = True
            w.start()

        log.debug("Sending work packets")
        in_queue_t.daemon = True
        in_queue_t.start()

        # Collect work from async
        log.debug("Aggregating async results")
        terminals_collected = 0
        f = 0
        lt = t = time.time()
        while terminals_collected < len(workers):
            packet = out_q.get()
            if packet is None:
                terminals_collected += 1
            elif isinstance(packet, Exception):
                raise packet
            else:
                r, v = packet
                mat[r] = v

                f += 1
                if report_interval and time.time() - lt >= report_interval:
                    log.debug("Rows per second: %f, Total: %d",
                              f / (time.time() - t), f)
                    lt = time.time()

        # All work should be exhausted at this point
        if use_multiprocessing and sys.platform == 'darwin':
            # multiprocessing.Queue.qsize doesn't work on OSX
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

        return mat
    finally:
        log.debug("Stopping/Joining queue feeder thread")
        in_queue_t.stop()
        in_queue_t.join()

        if use_multiprocessing:
            # Forcibly terminate worker processes if still alive
            log.debug("Joining/Terminating process workers")
            for w in workers:
                if w.is_alive():
                    w.terminate()
                w.join()

            log.debug("Cleaning multiprocess queues")
            for q in (in_q, out_q):
                q.close()
                q.join_thread()
        else:
            log.debug("Stopping/Joining threaded workers")
            for w in workers:
                w.stop()
                # w.join()
                # Threads should exit fine from here

        log.debug("Done")


class _FeedQueueThread (SmqtkObject, threading.Thread):

    def __init__(self, descr_elements, q, out_mat, num_terminal_packets):
        super(_FeedQueueThread, self).__init__()

        self.num_terminal_packets = num_terminal_packets
        self.out_mat = out_mat
        self.q = q
        self.descr_elements = descr_elements

        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.isSet()

    def run(self):
        try:
            # Special case for in-memory storage of descriptors
            from smqtk.representation.descriptor_element.local_elements \
                import DescriptorMemoryElement

            for r, d in enumerate(self.descr_elements):
                # If we've run out of matrix to fill,
                if r >= self.out_mat.shape[0]:
                    break

                if isinstance(d, DescriptorMemoryElement):
                    self.out_mat[r] = d.vector()
                else:
                    self.q.put((r, d))

                # If we're told to stop, immediately quit out of processing
                if self.stopped():
                    break
        except KeyboardInterrupt:
            pass
        except Exception as ex:
            self._log.error("Feeder thread encountered an exception: %s",
                            str(ex))
            self.q.put(ex)
        finally:
            self._log.debug("Sending in-queue terminal packets")
            for _ in range(self.num_terminal_packets):
                self.q.put(None)
            self._log.debug("Closing in-queue")


class _ElemVectorExtractorProcess (SmqtkObject, multiprocessing.Process):
    """
    Helper process for extracting DescriptorElement vectors on a separate
    process. This terminates with a None packet fed to in_q. Otherwise, in_q
    values are expected to be (row, element) pairs. Tuples of the form
    (row, vector) are published to the out_q.

    Terminal value: None

    """

    def __init__(self, i, in_q, out_q):
        super(_ElemVectorExtractorProcess, self)\
            .__init__(name='[w%d]' % i)
        self._log.debug("Making process worker (%d, %s, %s)", i, in_q, out_q)
        self.i = i
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        try:
            packet = self.in_q.get()
            while packet is not None:
                if isinstance(packet, Exception):
                    self.out_q.put(packet)
                else:
                    row, elem = packet
                    v = elem.vector()
                    self.out_q.put((row, v))
                packet = self.in_q.get()
            self.out_q.put(None)
        except KeyboardInterrupt:
            pass
        except Exception as ex:
            self._log.error("%s%s encountered an exception: %s",
                            self.__class__.__name__, self.name,
                            str(ex))
            self.out_q.put(ex)


class _ElemVectorExtractorThread (SmqtkObject, threading.Thread):
    """
    Helper process for extracting DescriptorElement vectors on a separate
    process. This terminates with a None packet fed to in_q. Otherwise, in_q
    values are expected to be (row, element) pairs. Tuples of the form
    (row, vector) are published to the out_q.

    Terminal value: None

    """

    def __init__(self, i, in_q, out_q, q_put_interval=0.001):
        SmqtkObject.__init__(self)
        threading.Thread.__init__(self, name='[w%d]' % i)

        self._log.debug("Making thread worker (%d, %s, %s)", i, in_q, out_q)
        self.i = i
        self.in_q = in_q
        self.out_q = out_q
        self.q_put_interval = q_put_interval

        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.isSet()

    def run(self):
        try:
            packet = self.in_q.get()
            while packet is not None and not self.stopped():
                if isinstance(packet, Exception):
                    self.out_q.put(packet)
                else:
                    row, elem = packet
                    v = elem.vector()
                    self.q_put((row, v))
                packet = self.in_q.get()
            self.q_put(None)
        except KeyboardInterrupt:
            pass
        except Exception as ex:
            self._log.error("%s%s encountered an exception: %s",
                            self.__class__.__name__, self.name,
                            str(ex))
            self.out_q.put(ex)

    def q_put(self, val):
        """
        Try to put the given value into the output queue until it is inserted
        (if it was previously full), or the stop signal was given.
        """
        put = False
        while not put and not self.stopped():
            try:
                self.out_q.put(val, timeout=self.q_put_interval)
                put = True
            except queue.Full:
                # self._log.debug("Skipping q.put Full error")
                pass
