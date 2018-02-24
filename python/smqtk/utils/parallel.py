import collections
import heapq
import logging
import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import sys
import threading
import traceback

from smqtk.utils import SmqtkObject
from six.moves import range, zip
from six.moves import zip_longest
from six.moves import queue


def parallel_map(work_func, *sequences, **kwargs):
    """
    Generalized local parallelization helper for executing embarrassingly
    parallel functions on an iterable of input data. This function then yields
    work results for data, optionally in the order that they were provided to
    this function.

    By default, we act like ``itertools.izip`` in regards to input sequences,
    whereby we stop performing work as soon as one of the input sequences is
    exhausted. The optional keyword argument ``fill_void`` may be specified to
    enable sequence handling like ``itertools.zip_longest`` where the longest
    sequence determines what is iterated, and the value given to ``fill_void``
    is used as the fill value.

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

    Input data given to ``sequences`` must be picklable in order to transport
    to worker threads/processes.

    :param work_func:
        Function that performs some work on input data, resulting in some
        returned value.

        When in multiprocessing mode, this cannot be a local function or a
        transport error will occur when trying to move the function to the
        worker process.
    :type work_func: (object, ...)-> object

    :param sequences: Input data to apply to the given ``work_func`` function.
        If more than one sequence is given, the function is called with an
        argument list consisting of the corresponding item of each sequence.
    :type sequences: collections.Iterable[collections.Iterable]

    :param kwargs: Optionally available keyword arguments are as follows:

        - fill_void
            - Optional value that, if specified, activates sequence handling
              like that of ``__builtin__.map`` except that the value provided
              is used to

        - ordered
            - If results for input elements should be yielded in the same order
              as input elements. If False, we yield results as soon as they are
              collected.
            - type: bool
            - default: False

        - buffer_factor
            - Multiplier against the number of processes used to limit the
              growth size of the result queue coming from worker processes
              (``int(cores * buffer_factor)``). This is utilized so we don't
              overrun our RAM buffering results.
            - type: float
            - default: 2.0

        - cores
            - Optional specification of the number of threads/cores to use. If
              None, we will attempt to use all available threads/cores.
            - type: None | int | long
            - default: None

        - use_multiprocessing
            - Whether or not to use discrete processes as the parallelization
              agent vs python threads.
            - type: bool
            - default: False

        - heart_beat
            - Interval at which workers check for operational messages while
              waiting on locks (e.g. waiting to push or pull messages). This
              ensures that workers are not left hanging, or hang the program,
              when and error or interruption occurs, or when waiting on an full
              edge. This must be >0.
            - type: float
            - default: 0.001

        - name
            - Optional string name for identifying workers and logging
              messages. ``None`` means no names are added.
            - type: str
            - default: None
    :type kwargs: dict

    :return: A new parallel results iterator that starts work on the input
        iterable when iterated.
    :rtype: ParallelResultsIterator


    Example
    -------
    >>> import math
    >>> result_iter = parallel_map(math.factorial, range(10), use_multiprocessing=True)
    >>> sorted(result_iter)
    [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
    """
    # kwargs
    cores = kwargs.get('cores', None)
    ordered = kwargs.get('ordered', False)
    buffer_factor = kwargs.get('buffer_factor', 2.0)
    use_multiprocessing = kwargs.get('use_multiprocessing', False)
    heart_beat = kwargs.get('heart_beat', 0.001)
    fill_activate = 'fill_void' in kwargs
    fill_value = kwargs.get('fill_void', None)
    name = kwargs.get('name', None)

    if name:
        log = logging.getLogger(__name__ + '[%s]' % name)
    else:
        log = logging.getLogger(__name__)

    if heart_beat <= 0:
        raise ValueError("heart_beat must be >0.")

    if cores is None or cores <= 0:
        cores = multiprocessing.cpu_count()
        log.debug("Using all cores (%d)", cores)
    else:
        log.debug("Only using %d cores", cores)

    # Choose parallel types
    if use_multiprocessing:
        queue_t = multiprocessing.Queue
        worker_t = _WorkerProcess
    else:
        queue_t = queue.Queue
        worker_t = _WorkerThread

    queue_work = queue_t(int(cores * buffer_factor))
    queue_results = queue_t(int(cores * buffer_factor))

    log.log(1, "Constructing worker processes")
    workers = [worker_t(name, i, work_func, queue_work, queue_results,
                        heart_beat)
               for i in range(cores)]

    log.log(1, "Constructing feeder thread")
    feeder_thread = _FeedQueueThread(name, sequences, queue_work,
                                     len(workers), heart_beat, fill_activate,
                                     fill_value)

    return ParallelResultsIterator(name, ordered, use_multiprocessing,
                                   heart_beat, queue_work,
                                   queue_results, feeder_thread, workers)


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


class ParallelResultsIterator (SmqtkObject, collections.Iterator):

    def __init__(self, name, ordered, is_multiprocessing, heart_beat,
                 work_queue, results_queue,
                 feeder_thread, workers):
        """
        :type ordered: bool
        :type is_multiprocessing: bool
        :type heart_beat: float
        :type work_queue: Queue.Queue | multiprocessing.queues.Queue
        :type results_queue: Queue.Queue | multiprocessing.queues.Queue
        :type feeder_thread: _FeedQueueThread
        :type workers: list[_WorkerThread|_WorkerProcess]
        """
        if name:
            self.name = '[' + name + ']'
        else:
            self.name = ''

        self.ordered = ordered
        if self.ordered:
            self._log.debug("Maintaining result iteration order based on input "
                            "order")
        self.heart_beat = heart_beat
        self.is_multiprocessing = is_multiprocessing

        self.work_queue = work_queue
        self.results_queue = results_queue
        self.feeder_thread = feeder_thread
        self.workers = workers

        self.has_started_workers = False
        self.has_cleaned_up = False

        self.found_terminals = 0
        self.result_heap = []
        self.next_index = 0

        self.stop_event = threading.Event()
        self.stop_event_lock = threading.Lock()

    @property
    def _log(self):
        # Changing naming of logger returned from default
        return logging.getLogger(
            self.get_logger().name + self.name
        )

    def __repr__(self):
        sfx = ''
        if self.name:
            sfx = '[' + self.name + ']'
        return "<%(module)s.%(class)s%(sfx)s at %(address)s>" % {
            "module": self.__module__,
            "class": self.__class__.__name__,
            "sfx": sfx,
            "address": hex(id(self)),
        }

    def __next__(self):
        try:
            if not self.has_started_workers:
                self.start_workers()

            while (self.found_terminals < len(self.workers) and
                   not self.stopped()):
                packet = self.results_q_get()

                if is_terminal(packet):
                    self._log.log(1, 'Found terminal')
                    self.found_terminals += 1
                elif isinstance(packet[0], Exception):
                    ex, formatted_exc = packet
                    self._log.warn('Received exception: {}\n{}'.format(
                            ex, formatted_exc))
                    raise ex
                else:
                    i, result = packet
                    if self.ordered:
                        heapq.heappush(self.result_heap, (i, result))
                        if self.result_heap[0][0] == self.next_index:
                            _, result = heapq.heappop(self.result_heap)
                            self.next_index += 1
                            return result
                    else:
                        return result

            # Go through heap if there's anything in it
            if self.result_heap:
                _, result = heapq.heappop(self.result_heap)
                return result

            # Nothing left
            if not self.stopped():
                self._log.log(1, "Asserting empty queues on what looks like a "
                                 "full iteration.")
                self.assert_queues_empty()

            raise StopIteration()

        except Exception:
            self.stop()
            raise

    next = __next__

    def start_workers(self):
        """
        Start worker threads/processes
        """
        self._log.log(1, "Starting worker processes")
        for w in self.workers:
            w.start()

        self._log.log(1, "Starting feeder thread")
        # self.feeder_thread.daemon = True
        self.feeder_thread.start()

        self.has_started_workers = True

    def clean_up(self):
        """
        Clean up any live resources if we haven't done so already.
        """
        if self.has_started_workers and not self.has_cleaned_up:
            self._log.log(1, "Stopping feeder thread")
            self.feeder_thread.stop()
            self.feeder_thread.join()

            self._log.log(1, "Stopping workers")
            for w in self.workers:
                w.stop()
                w.join()

            if self.is_multiprocessing:
                self._log.log(1, "Closing/Joining process queues")
                for q in (self.work_queue, self.results_queue):
                    q.close()
                    q.join_thread()

            self.has_cleaned_up = True

    def stop(self):
        """
        Stop this iterator.

        This does not clean up resources (see ``clean_up`` for that).
        """
        with self.stop_event_lock:
            self.stop_event.set()
            self.clean_up()

    def stopped(self):
        """
        :return: if this iterator has been stopped
        :rtype: bool
        """
        return self.stop_event.is_set()

    def results_q_get(self):
        """
        Attempts to get something from the results queue.

        :raises StopIteration: when we've been told to stop.

        """
        while not self.stopped():
            try:
                return self.results_queue.get(timeout=self.heart_beat)
            except queue.Empty:
                pass
        raise StopIteration()

    def assert_queues_empty(self):
        # All work should be exhausted at this point
        if self.is_multiprocessing and sys.platform == 'darwin':
            # multiprocessing.Queue.qsize doesn't work on OSX
            # - Try to get something from each queue, expecting an empty
            #   exception.
            try:
                self.work_queue.get(block=False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise AssertionError("In queue not empty")
            try:
                self.results_queue.get(block=False)
            except multiprocessing.queues.Empty:
                pass
            else:
                raise AssertionError("Out queue not empty")
        else:
            assert self.work_queue.qsize() == 0, \
                "In queue not empty (%d)" % self.work_queue.qsize()
            assert self.results_queue.qsize() == 0, \
                "Out queue not empty (%d)" % self.results_queue.qsize()


class _FeedQueueThread (SmqtkObject, threading.Thread):
    """
    Helper thread for putting data into the work queue

    """

    def __init__(self, name, arg_sequences, q, num_terminal_packets, heart_beat,
                 do_fill, fill_value):
        threading.Thread.__init__(self, name=name)
        SmqtkObject.__init__(self)

        if name:
            self.name = '[' + name + ']'
        else:
            self.name = ''

        self.arg_sequences = arg_sequences
        self.q = q
        self.num_terminal_packets = num_terminal_packets
        self.heart_beat = heart_beat
        self.do_fill = do_fill
        self.fill_value = fill_value

        self._stop_event = threading.Event()

        # self.daemon = True

    @property
    def _log(self):
        # Changing naming of logger returned from default
        return logging.getLogger(
            self.get_logger().name + self.name
        )

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.isSet()

    def run(self):
        self._log.log(1, "Starting")

        if self.do_fill:
            _zip = zip_longest
            _zip_kwds = {'fillvalue': self.fill_value}
        else:
            _zip = zip
            _zip_kwds = {}

        try:
            r = 0
            for args in _zip(*self.arg_sequences, **_zip_kwds):
                self.q_put((r, args))
                r += 1

                # If we're told to stop, immediately quit out of processing
                if self.stopped():
                    self._log.log(1, "Told to stop prematurely")
                    break
        except Exception as ex:
            self._log.warn("Caught exception %s", type(ex))
            self.q_put((ex, traceback.format_exc()))
            self.stop()
        else:
            self._log.log(1, "Sending in-queue terminal packets")
            for _ in range(self.num_terminal_packets):
                self.q_put(_TerminalPacket())
        finally:
            # Explicitly stop any nested parallel maps
            for s in self.arg_sequences:
                if isinstance(s, ParallelResultsIterator):
                    self._log.log(1, "Stopping nested parallel map: %s", s)
                    s.stop()

            self._log.log(1, "Closing")

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
            except queue.Full:
                pass


class _Worker (SmqtkObject):

    def __init__(self, name, i, work_function, in_q, out_q, heart_beat):
        """
        :type name: str
        :type i: int
        :type work_function: (*args) -> object
        :type in_q: multiprocessing.queues.Queue
        :type out_q: multiprocessing.queues.Queue
        :type heart_beat: float
        """
        if name:
            self.name = '[' + name + '::%d]' % i
        else:
            self.name = '::%d' % i

        self.i = i
        self.work_function = work_function
        self.in_q = in_q
        self.out_q = out_q
        self.heart_beat = heart_beat
        self._log.log(1, "Making process worker (%d, %s, %s)", i, in_q, out_q)

        self._stop_event = self._make_event()

    @property
    def _log(self):
        return logging.getLogger(
            self.get_logger().name + self.name
        )

    def _make_event(self):
        raise NotImplementedError()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        try:
            packet = self.q_get()
            while not self.stopped():
                if is_terminal(packet):
                    self._log.log(1, "sending terminal")
                    self.q_put(packet)
                    self.stop()
                elif isinstance(packet[0], Exception):
                    # Pass exception along
                    self.q_put(packet)
                    self.stop()
                else:
                    i, args = packet
                    result = self.work_function(*args)
                    self.q_put((i, result))
                    packet = self.q_get()
        # Transport back any exceptions raised
        except Exception as ex:
            self._log.warn("Caught exception %s", type(ex))
            self.q_put((ex, traceback.format_exc()))
            self.stop()
        finally:
            self._log.log(1, "Closing")

    def q_get(self):
        """
        Try to get a value from the queue while keeping an eye out for an exit
        request.

        :return: next value on the input queue

        """
        while not self.stopped():
            try:
                return self.in_q.get(timeout=self.heart_beat)
            except queue.Empty:
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
            except queue.Full:
                pass


class _WorkerProcess (_Worker, multiprocessing.Process):

    def __init__(self, name, i, work_function, in_q, out_q, heart_beat):
        multiprocessing.Process.__init__(self)
        _Worker.__init__(self, name, i, work_function, in_q, out_q, heart_beat)
        # if name:
        #     self.name = name + "::%d" % i

        # self.daemon = True

    def _make_event(self):
        return multiprocessing.Event()

    run = _Worker.run


class _WorkerThread (_Worker, threading.Thread):

    def __init__(self, name, i, work_function, in_q, out_q, heart_beat):
        threading.Thread.__init__(self)
        _Worker.__init__(self, name, i, work_function, in_q, out_q, heart_beat)
        # if name:
        #     self.name = name + "::%d" % i

        # self.daemon = True

    def _make_event(self):
        return threading.Event()

    run = _Worker.run
