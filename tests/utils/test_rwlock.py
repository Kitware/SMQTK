from collections import OrderedDict

import threading
import time
import unittest

from smqtk.utils.read_write_lock import \
    ContextualReadWriteLock


def wait_for_value(f, timeout):
    """
    Wait a specified timeout period of time (seconds) for the given
    function to execute successfully.

    `f` usually wraps an assertion function.

    :param f: Assertion function.
    :type f: () -> None

    :param timeout: Time out in seconds to wait for convergence.
    :type timeout: float
    """
    s = time.time()
    neq = True
    while neq:
        try:
            f()
            # function passed.
            neq = False
        except (Exception, AssertionError):
            # if assertion fails past timeout, actually raise assertion.
            if time.time() - s > timeout:
                raise


class TestContextualReadWriteLock (unittest.TestCase):

    def setUp(self):
        self.state = OrderedDict()

    def wait_for_state(self, k):
        """ Wait forever until a state attribute is True. """
        while k not in self.state or not self.state[k]:
            pass

    # Added asserts

    def assertInState(self, k):
        """ Assert key in state """
        self.assertIn(k, self.state)

    def assertLockFree(self, l):
        self.assertEqual(l._semlock._get_value(), 1)

    def assertLockAcquired(self, l):
        self.assertEqual(l._semlock._get_value(), 0)

    # Unit Tests

    def test_initial_state(self):
        # Test expected lock and value states before use.
        crwl = ContextualReadWriteLock()
        self.assertLockFree(crwl._service_lock)
        self.assertLockFree(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

    def test_read_context_state(self):
        # Test expected state conditions when transitioning into and out of a
        # read-lock context.
        crwl = ContextualReadWriteLock()

        def t1(c):
            with c.read_context():
                self.state['t1_read_acquired'] = True
                self.wait_for_state('t1_release')
            self.state['t1_read_released'] = True

        t1 = threading.Thread(target=t1, args=(crwl,))
        t1.daemon = True
        t1.start()

        # Thread should immediately attempt to acquire read lock.  We should see
        # that it does successfully.
        wait_for_value(lambda: self.assertInState('t1_read_acquired'),
                       1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 1)

        # Trigger thread to release context and check state.
        self.state['t1_release'] = True
        wait_for_value(lambda: self.assertInState('t1_read_released'),
                       1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockFree(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

    def test_write_context_state(self):
        # Test expected state conditions when transitioning into and out of a
        # write-lock context.
        crwl = ContextualReadWriteLock()

        def t1_func(c):
            with c.write_context():
                self.state['t1_write_acquired'] = True
                self.wait_for_state('t1_release')
            self.state['t1_write_released'] = True

        t1 = threading.Thread(target=t1_func, args=(crwl,))
        t1.daemon = True
        t1.start()

        # Thread should immediately attempt to acquire write lock.  We should
        # see that it does successfully.
        wait_for_value(lambda: self.assertInState('t1_write_acquired'),
                       1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

        # Trigger thread to release context and check state.
        self.state['t1_release'] = True
        wait_for_value(lambda: self.assertInState('t1_write_released'),
                       1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockFree(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

    def test_concurrent_read_then_write(self):
        # Test that a thread with a read lock blocks a write lock from entering.
        crwl = ContextualReadWriteLock()

        # Thread 1 function - Read lock
        def t1_func(c):
            with c.read_context():
                self.state['t1_read_acquired'] = True
                self.wait_for_state('t1_release')
            self.state['t1_read_released'] = True

        # Thread 2 function - Write lock
        def t2_func(c):
            self.wait_for_state('t2_acquire')
            with c.write_context():
                self.state['t2_write_acquired'] = True
                self.wait_for_state('t2_release')
            self.state['t2_write_released'] = True

        t1 = threading.Thread(target=t1_func, args=(crwl,))
        t2 = threading.Thread(target=t2_func, args=(crwl,))
        t1.daemon = t2.daemon = True
        t1.start()
        t2.start()

        # Upon starting threads, t1 should get read lock and t2 should not have
        # done anything yet.
        wait_for_value(lambda: self.assertInState('t1_read_acquired'), 1.0)
        self.assertNotIn('t2_write_acquired', self.state)
        self.assertLockFree(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 1)

        # t2 should attempt to acquire write context but be blocked.  We should
        # see that the service lock is acquired and that 't2_write_acquired' is
        # not set.
        self.state['t2_acquire'] = True
        wait_for_value(lambda: self.assertLockAcquired(crwl._service_lock), 1.0)
        self.assertNotIn('t2_write_acquired', self.state)
        self.assertLockAcquired(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 1)

        # Releasing t1's read lock should cause t2 to acquire write lock.
        self.state['t1_release'] = True
        wait_for_value(lambda: self.assertInState('t1_read_released'), 1.0)
        wait_for_value(lambda: self.assertInState('t2_write_acquired'), 1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

        # t2 should now be able to release the write lock like normal
        self.state['t2_release'] = True
        wait_for_value(lambda: self.assertInState('t2_write_released'), 1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockFree(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

    def test_concurrent_write_then_read(self):
        # Test that a thread with a read lock blocks a write lock from entering.
        crwl = ContextualReadWriteLock()

        # Thread 1 function - Write lock
        def t1_func(c):
            with c.write_context():
                self.state['t1_write_acquired'] = True
                self.wait_for_state('t1_release')
            self.state['t1_write_released'] = True

        # Thread 2 function - Read lock
        def t2_func(c):
            self.wait_for_state('t2_acquire')
            self.state['t2_read_attempt'] = True
            with c.read_context():
                self.state['t2_read_acquired'] = True
                self.wait_for_state('t2_release')
            self.state['t2_read_released'] = True

        t1 = threading.Thread(target=t1_func, args=(crwl,))
        t2 = threading.Thread(target=t2_func, args=(crwl,))
        t1.daemon = t2.daemon = True
        t1.start()
        t2.start()

        # Upon starting threads, t1 should get write lock and t2 should not have
        # done anything yet.
        wait_for_value(lambda: self.assertInState('t1_write_acquired'), 1.0)
        self.assertNotIn('t2_read_acquired', self.state)
        self.assertLockFree(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

        # t2 should attempt to acquire read context but be blocked.  We should
        # see that the service lock is acquired and that 't2_read_acquired' is
        # not set.
        self.state['t2_acquire'] = True
        wait_for_value(lambda: self.assertLockAcquired(crwl._service_lock), 1.0)
        self.assertNotIn('t2_write_acquired', self.state)
        self.assertLockAcquired(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockAcquired(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)

        # Releasing t1's write lock should cause t2 to acquire read lock.
        self.state['t1_release'] = True
        wait_for_value(lambda: self.assertInState('t1_write_released'), 1.0)
        wait_for_value(lambda: self.assertInState('t2_read_acquired'), 1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockAcquired(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 1)

        # t2 should now be able to release the read lock like normal
        self.state['t2_release'] = True
        wait_for_value(lambda: self.assertInState('t2_read_released'), 1.0)
        self.assertLockFree(crwl._service_lock)
        self.assertLockFree(crwl._resource_lock)
        self.assertLockFree(crwl._reader_count_lock)
        self.assertEqual(crwl._reader_count, 0)
