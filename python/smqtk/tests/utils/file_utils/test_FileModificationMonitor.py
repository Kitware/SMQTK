from __future__ import print_function

import atexit
import os
import tempfile
import time
import threading
import unittest

import smqtk.utils.file_utils


class TestFileModificationMonitor (unittest.TestCase):

    @staticmethod
    def _mk_test_fp():
        fd, fp = tempfile.mkstemp()
        os.close(fd)
        atexit.register(lambda: os.remove(fp))
        return fp

    def test_monitor_stop(self):
        # test that monitor stops when its told to
        fp = self._mk_test_fp()

        has_triggered = [False]

        def cb(filepath):
            has_triggered[0] = True
            self.assertEqual(filepath, fp)

        interval = 0.01
        monitor = smqtk.utils.file_utils.FileModificationMonitor(fp, interval,
                                                                 0.5, cb)
        self.assertTrue(monitor.stopped())

        monitor.start()

        try:
            self.assertFalse(has_triggered[0])
            self.assertTrue(monitor.is_alive())
            self.assertFalse(monitor.stopped())

            monitor.stop()
            # If thread hasn't entered while loop yet, it will immediately kick
            # out, otherwise its sleeping for the given interval.
            monitor.join(interval*2)

            self.assertFalse(has_triggered[0])
            self.assertFalse(monitor.is_alive())
        finally:
            if monitor.is_alive():
                print("WARNING :: Forcing thread stop by removing filepath var")
                monitor.filepath = None

    def test_short_file_copy(self):
        # where "short" means effectively instantaneous file creation / copy
        # / touch.
        #
        # procedure:
        #   - create a file via mkstemp
        #   - create file monitor with detection callback and non-zero settle
        #       time.
        #   - touch file
        #   - check that callback was NOT triggered immediately
        #   - wait settle time / 2, check that cb NOT triggered yet
        #   - wait settle time / 4, check that cb NOT triggered yet
        #   - wait settle time / 4, check that cb HAS been called.

        fp = self._mk_test_fp()

        has_triggered = [False]

        def cb(filepath):
            has_triggered[0] = True
            self.assertEqual(filepath, fp)

        interval = 0.01
        settle = 0.1
        monitor = smqtk.utils.file_utils.FileModificationMonitor(fp, interval,
                                                                 settle, cb)
        try:
            monitor.start()
            # file not touched, should still be waiting
            self.assertEqual(monitor.state, monitor.STATE_WAITING)
            self.assertFalse(has_triggered[0])

            time.sleep(interval)
            smqtk.utils.file_utils.touch(fp)
            time.sleep(interval*2)
            monitor._log.info('checking')
            self.assertFalse(has_triggered[0])
            self.assertEqual(monitor.state, monitor.STATE_WATCHING)

            time.sleep(settle / 2.)
            monitor._log.info('checking')
            self.assertEqual(monitor.state, monitor.STATE_WATCHING)
            self.assertFalse(has_triggered[0])

            time.sleep(settle / 4.)
            monitor._log.info('checking')
            self.assertEqual(monitor.state, monitor.STATE_WATCHING)
            self.assertFalse(has_triggered[0])

            time.sleep(settle / 4.)
            monitor._log.info('checking')
            self.assertTrue(has_triggered[0])

        finally:
            monitor.stop()
            monitor.join()

    def test_long_file_wait(self):
        # procedure:
        #   - create a file via mkstemp
        #   - create file monitor with detection callback and non-zero settle
        #       time.
        #   - setup/start thread that appends to file at an interval that is
        #       less than settle time
        #   - wait and check that cb hasn't been called a few times
        #   - stop appender thread
        #   - check that cb called after settle period

        fp = self._mk_test_fp()

        has_triggered = [False]
        append_interval = 0.02
        monitor_interval = 0.01
        monitor_settle = 0.1

        def cb(filepath):
            has_triggered[0] = True
            self.assertEqual(filepath, fp)

        class AppendThread (threading.Thread):
            def __init__(self):
                super(AppendThread, self).__init__()
                self._s = threading.Event()

            def stop(self):
                self._s.set()

            def stopped(self):
                return self._s.is_set()

            def run(self):
                while not self.stopped():
                    with open(fp, 'a') as f:
                        f.write('0')
                    time.sleep(append_interval)

        m_thread = smqtk.utils.file_utils.FileModificationMonitor(
            fp, monitor_interval, monitor_settle, cb
        )
        a_thread = AppendThread()

        try:
            m_thread.start()
            a_thread.start()

            time.sleep(monitor_settle)
            m_thread._log.info('checking')
            self.assertFalse(m_thread.stopped())
            self.assertFalse(has_triggered[0])
            self.assertEqual(m_thread.state, m_thread.STATE_WATCHING)

            time.sleep(monitor_settle)
            m_thread._log.info('checking')
            self.assertFalse(m_thread.stopped())
            self.assertFalse(has_triggered[0])
            self.assertEqual(m_thread.state, m_thread.STATE_WATCHING)

            a_thread.stop()

            time.sleep(monitor_settle)
            m_thread._log.info('checking')
            self.assertTrue(has_triggered[0])

        finally:
            a_thread.stop()
            m_thread.stop()

    def test_invalid_params(self):
        fp = self._mk_test_fp()

        # Invalid path value
        self.assertRaises(
            ValueError,
            smqtk.utils.file_utils.FileModificationMonitor,
            '/not/real', 1, 1, lambda p: None
        )
        # Invalid timers values
        self.assertRaises(
            ValueError,
            smqtk.utils.file_utils.FileModificationMonitor,
            fp, -1, 1, lambda p: None
        )
        self.assertRaises(
            ValueError,
            smqtk.utils.file_utils.FileModificationMonitor,
            fp, 1, -1, lambda p: None
        )
