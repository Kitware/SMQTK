# -*- coding: utf-8 -*-

import time


class SimpleTimer (object):
    """
    Little class to wrap the timing of things. To be use with the ``with``
    statement.
    """

    def __init__(self, msg, logger=None):
        self._log = logger
        self._msg = msg
        self._s = 0.0

    def __enter__(self):
        if self._log:
            self._log.info(self._msg)
        else:
            print "[SimpleTimer]", self._msg
        self._s = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._log:
            self._log.info("-> %f s", time.time() - self._s)
        else:
            print "[SimpleTimer] -> %f s" % (time.time() - self._s)
