# -*- coding: utf-8 -*-

import time


class SimpleTimer (object):
    """
    Little class to wrap the timing of things. To be use with the ``with``
    statement.
    """

    def __init__(self, msg, log_func=None, *args):
        """
        Additional arguments are passed to the logging method
        :param msg:
        :param log_func:
        :param args:
        :return:
        """
        self._log_func = log_func
        self._msg = msg
        self._msg_args = args
        self._s = 0.0

    def __enter__(self):
        if self._log_func:
            self._log_func(self._msg, *self._msg_args)
        else:
            print "[SimpleTimer]", self._msg % self._msg_args
        self._s = time.time()

    def __exit__(self, *_):
        if self._log_func:
            self._log_func("%s -> %f s", self._msg % self._msg_args,
                           time.time() - self._s)
        else:
            print "[SimpleTimer] %s -> %f s" % (self._msg % self._msg_args,
                                                time.time() - self._s)
