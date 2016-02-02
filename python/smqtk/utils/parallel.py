import heapq
import logging
import multiprocessing
import multiprocessing.queues
import Queue
import sys
import threading

from smqtk.utils import SmqtkObject


__author__ = "paul.tunison@kitware.com"


def parallel_map(data_iter, work_func,
                 procs=None, ordered=False, buffer_factor=2,
                 use_multiprocessing=False, heart_beat=0.001):
    """
    Generalized local parallelization helper for executing embarrassingly
    parallel functions on an iterable of input data. This function then yields
    work results for data, optionally the order that they were provided to this
    function.

    This is intended to be able to replace ``multiprocessing.pool.Pool`` and
    ``multiprocessing.pool.ThreadPool`` uses with the added benefit of:
        - No set-up or clean-up needed
        - No performance loss compared to ``multiprocessing.pool`` classes
          for non-trivial work functions (like IO operations).
        - We iterate results as they are ready (optionally in order of
          input)
        - Lambda or on-the-fly function can be provided as the work function
          when using multiprocessing.
        - Buffered input/output queues so that mapping work of a very large
          input set doesn't overrun your memory (e.g. iterating over many large
          vectors/matrices).

    This function is, however, slower than multiprocessing.pool classes for
    trivial functions, like using the function ``ord`` over a set of
    characters.

    Input data from ``data_iter`` must be picklable in order to transport to
    worker threads/processes.

    :TODO: This currently only works for functions that take a single parameter
    but can be fairly easily extended to take multiple argument sequences +
    an option on whether to stop on the shorted or longest sequence (i.e. map
    vs. imap semantics)

    :param data_iter: Input data to map to the given ``work_func``.
    :type data_iter: collections.Iterable

    :param work_func:
        Function that performs some work on input data, resulting in some
        returned value.

        When in multiprocessing mode, this cannot be a local function or a
        transport error will occur when trying to move the function to the
        worker process.
    :type work_func: (object)-> object

    :param procs: Optional specification of the number of threads/cores to use.
        If None, we will attempt to use all available threads/cores.
    :type procs: None | int | long

    :param ordered: If results for input elements should be returned should be
        in the same order as input elements. If False, we yield results as soon
        as they are collected.
    :type ordered: bool

    :param buffer_factor: Multiplier against the number of processes used to
        limit the growth size of the result queue coming from worker processes
        (``int(procs * buffer_factor)``). This is utilized so we don't overrun
        our RAM buffering results.
    :type buffer_factor: float

    :param use_multiprocessing: Whether or not to use discrete processes as the
        parallelization agent vs python threads.
    :type use_multiprocessing: bool

    :param heart_beat: Interval at which workers check for operational messages
        while waiting on locks (e.g. waiting to push or pull messages). This
        ensures that workers are not left hanging, or hang the program, when
        and error or interruption occurs, or when waiting on an full edge. This
        must be >0.
    :type heart_beat: float

    """
    log = logging.getLogger(__name__)

    if heart_beat <= 0:
        raise ValueError("heart_beat must be >0.")

    if not procs or procs <= 0:
        procs = multiprocessing.cpu_count()
        log.debug("Using all cores (%d)", procs)
    else:
        log.debug("Only using %d cores", procs)

    # Choose parallel types
    hb_arg = {}
    if use_multiprocessing:
        queue_t = multiprocessing.queues.Queue
        worker_t = _WorkProcess
    else:
        queue_t = Queue.Queue
        worker_t = _WorkThread
        hb_arg['heart_beat'] = heart_beat

    queue_work = queue_t(int(procs * buffer_factor))
    queue_results = queue_t(int(procs * buffer_factor))

    log.debug("Constructing worker processes")
    workers = [worker_t(i, work_func, queue_work, queue_results, **hb_arg)
               for i in range(procs)]

    log.debug("Constructing feeder thread")
    feeder_thread = _FeedQueueThread(data_iter, queue_work, len(workers),
                                     heart_beat)

    try:
        log.debug("Starting worker processes")
        for w in workers:
            w.start()

        log.debug("Starting feeder thread")
        feeder_thread.start()

        # Collect/Yield work results
        found_terminals = 0
        heap = []
        next_index = 0
        while found_terminals < len(workers):
            packet = queue_results.get()

            if is_terminal(packet):
                found_terminals += 1
            elif isinstance(packet, Exception):
                raise packet
            else:
                i, result = packet
                if ordered:
                    heapq.heappush(heap, (i, result))
                    if heap[0][0] == next_index:
                        _, result = heapq.heappop(heap)
                        yield result
                        next_index += 1
                else:
                    yield result
        # If we're in ordered mode, there may still be things left in the heap
        # (received out of order).
        while heap:
            i, result = heapq.heappop(heap)
            yield result

        # Done performing work and collecting results at this point

        # All work should be exhausted at this point
        if use_multiprocessing and sys.platform == 'darwin':
            # multiprocessing.Queue.qsize doesn't work on OSX
            # - Try to get something from each queue, expecting an empty
            #   exception.
            try:
                queue_work.get(block=False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise AssertionError("In queue not empty")
            try:
                queue_results.get(block=False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise AssertionError("Out queue not empty")
        else:
            assert queue_work.qsize() == 0, \
                "In queue not empty (%d)" % queue_work.qsize()
            assert queue_results.qsize() == 0, \
                "Out queue not empty (%d)" % queue_results.qsize()
    finally:
        log.debug("Stopping feeder thread")
        feeder_thread.stop()
        feeder_thread.join()

        if use_multiprocessing:
            log.debug("Terminating/Joining worker processes")
            for w in workers:
                if w.is_alive():
                    w.terminate()
                w.join()

            log.debug("Cleaning up queues")
            for q in (queue_work, queue_results):
                q.close()
                q.join_thread()
        else:
            log.debug("Stopping worker threads")
            for w in workers:
                w.stop()

    log.debug("Done")


class _TerminalPacket (object):
    """
    Signals a terminal message
    """


def is_terminal(p):
    """
    Check if a given packet is a terminal element.

    :param p: element to check
    :type p: object

    :return: If ``p`` is a terminal element
    :rtype: bool

    """
    return isinstance(p, _TerminalPacket)


class _FeedQueueThread (SmqtkObject, threading.Thread):
    """
    Helper thread for putting data into the work queue

    """

    def __init__(self, data_iter, q, num_terminal_packets, heart_beat):
        super(_FeedQueueThread, self).__init__()

        self.data_iter = data_iter
        self.q = q
        self.num_terminal_packets = num_terminal_packets
        self.heart_beat = heart_beat

        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        try:
            for r, d in enumerate(self.data_iter):
                self.q_put((r, d))

                # If we're told to stop, immediately quit out of processing
                if self.stopped():
                    self._log.debug("Told to stop prematurely")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._log.debug("Sending in-queue terminal packets")
            for _ in xrange(self.num_terminal_packets):
                self.q_put(_TerminalPacket())
            self._log.debug("Closing in-queue")

    def q_put(self, val):
        """
        Try to put the given value into the output queue until it is inserted
        (if it was previously full), or the stop signal was given.

        :param val: value to put into the output queue.

        """
        put = False
        while not put and not self.stopped():
            try:
                self.q.put(val, timeout=self.heart_beat)
                put = True
            except Queue.Full:
                pass


class _WorkProcess (SmqtkObject, multiprocessing.Process):
    """
    Helper process for extracting DescriptorElement vectors on a separate
    process. This terminates with a None packet fed to in_q. Otherwise, in_q
    values are expected to be (row, element) pairs. Tuples of the form
    (row, vector) are published to the out_q.

    """

    def __init__(self, i, work_function, in_q, out_q):
        super(_WorkProcess, self).__init__(name='[w%d]' % i)

        self._log.debug("Making process worker (%d, %s, %s)", i, in_q, out_q)
        self.i = i
        self.work_function = work_function
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        try:
            packet = self.in_q.get()
            while not is_terminal(packet):
                i, data = packet
                result = self.work_function(data)
                self.out_q.put((i, result))
                packet = self.in_q.get()
        except KeyboardInterrupt:
            pass
        # Transport back any exceptions raised
        except Exception, ex:
            self.out_q.put(ex)
            raise
        finally:
            self._log.debug("%s finished work", self.name)
            self.out_q.put(_TerminalPacket())


class _WorkThread (SmqtkObject, threading.Thread):
    """
    Helper process for extracting DescriptorElement vectors on a separate
    process. This terminates with a None packet fed to in_q. Otherwise, in_q
    values are expected to be (row, element) pairs. Tuples of the form
    (row, vector) are published to the out_q.

    """

    def __init__(self, i, work_function, in_q, out_q, heart_beat):
        """
        :type i: int
        :type work_function:
        :type in_q: Queue.Queue
        :type out_q: Queue.Queue
        :type heart_beat: float
        """
        SmqtkObject.__init__(self)
        threading.Thread.__init__(self, name='[w%d]' % i)

        self._log.debug("Making thread worker (%d, %s, %s)", i, in_q, out_q)
        self.i = i
        self.work_function = work_function
        self.in_q = in_q
        self.out_q = out_q
        self.heart_beat = heart_beat

        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        try:
            packet = self.q_get()
            while not is_terminal(packet) and not self.stopped():
                i, data = packet
                result = self.work_function(data)
                self.q_put((i, result))
                packet = self.q_get()
        # Transport back any exceptions raised
        except Exception, ex:
            self.q_put(ex)
            raise
        finally:
            self._log.debug("%s finished work", self.name)
            self.q_put(_TerminalPacket())

    def q_get(self):
        """
        Try to get a value from the queue while keeping an eye out for an exit
        request.

        :return: next value on the input queue

        """
        while not self.stopped():
            try:
                return self.in_q.get(timeout=self.heart_beat)
            except Queue.Empty:
                pass

    def q_put(self, val):
        """
        Try to put the given value into the output queue while keeping an eye
        out for an exit request.

        :param val: value to put into the output queue.

        """
        put = False
        while not put and not self.stopped():
            try:
                self.out_q.put(val, timeout=self.heart_beat)
                put = True
            except Queue.Full:
                pass
