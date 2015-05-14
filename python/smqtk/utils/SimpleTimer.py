# -*- coding: utf-8 -*-

import time


class SimpleTimer (object):
    """
    Little class to wrap the timing of things. To be use with the ``with``
    statement.
    """

    def __init__(self, msg, log_func=None):
        self._log_func = log_func
        self._msg = msg
        self._s = 0.0

    def __enter__(self):
        if self._log_func:
            self._log_func(self._msg)
        else:
            print "[SimpleTimer]", self._msg
        self._s = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._log_func:
            self._log_func("%s -> %f s", self._msg, time.time() - self._s)
        else:
            print "[SimpleTimer] %s -> %f s" % (self._msg,
                                                time.time() - self._s)
