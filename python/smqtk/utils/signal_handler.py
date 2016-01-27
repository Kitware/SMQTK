"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Object for handling signal overrides and managing what signals have been seen
since override.

"""

import logging
import re
import signal
import threading


class SignalHandler(object):
    """
    Object for handling knowledge of whether a particular system signal has
    been caught for the current thread of execution.

    This object is thread safe and acts in a synchronous manner.
    """

    def __init__(self, name='SignalHandler'):
        self.__signal_caught = {}
        self.__s_lock = threading.Lock()
        self.__prev_handlers = {}
        self.__h_lock = threading.Lock()

        self._log = logging.getLogger(__name__ + '.' + name)

        # create a mapping of signal integer keys to the signal they represent
        # this may change from system to system(?), thus the creating it
        # automatically
        sig_list = [
            re.match('^SIG[A-Z]+$', e) and re.match('^SIG[A-Z]+$', e).group()
            for e in dir(signal)
        ]
        sig_list.sort()
        sig_list = sig_list[sig_list.count(None):]
        self._sig_map = dict([(getattr(signal, sig), sig)
                              for sig in sig_list])

    def _handle_signal(self, signum, stack):
        """
        Callback method to be registered to signal.signal for a particular
        signal to catch.
        """
        # print "Caught signal:", signum
        with self.__s_lock:
            self._log.debug("'%s' caught", self._sig_map[signum])
            self.__signal_caught[signum] = True

    def _gen_signal_handle(self, custom_func=None):
        """
        Generated a callback method to be registered to signal.signal for a
        particular signal to catch.

        :param custom_func: A custom signal handle function to add
            functionality to our handler.
        :type custom_func: (int, None|frame) -> None

        :return: Function handle method for registering.
        :rtype: types.FunctionType

        """
        def handle_signal(signum, stack):
            """
            Callback method to be registered to signal.signal for a particular
            signal to catch.

            :type signum: int
            :type stack: None | frame

            """
            with self.__s_lock:
                self._log.debug("'%s' caught", self._sig_map[signum])
                self.__signal_caught[signum] = True

            if custom_func:
                custom_func(signum, stack)

        return handle_signal

    def register_signal(self, signum, custom_func=None):
        """
        Register a signal to be handled and monitored if not already
        registered.
        We will override and record the previous handler so that we may put it
        back if/when unregister that signal.

        A custom handling function may be passed to extend actions taken when
        the signal is caught. This method should be of the form of a normal
        signal handle method (see python docs). Regardless if a custom function
        is given, we will always register in our map that a handler has been
        registered for that signal, preventing another handle to be registered
        for that signal until the current one is removed, as well as still
        registering that the signal has been caught.

        If we have already registered the given signal, we return False.

        :type signum: int
        :param signum: The identifying integer value of the signal to check.
        (use signal.SIGINT, signal.SIGTERM, etc.)

        :param custom_func: Custom callback function to be called when the
            specified signal is caught. This function must take two arguments
            where the first is the integer signal identifier and the second
            is the stack frame in which the signal was caught.
        :type custom_func: (int, frame) -> None

        :return: True of we successfully registered a new signal, or False if
        the signal is already registered.

        """
        # if we haven't already registered a handler for the given signal.
        with self.__h_lock:
            if signum not in self.__prev_handlers:
                self._log.debug("Registering catch for signal %i (%s)",
                                signum, self._sig_map[signum])
                prev_handle = \
                    signal.signal(signum, self._gen_signal_handle(custom_func))
                self.__prev_handlers[signum] = prev_handle
                return True
            self._log.debug("%s already registered", self._sig_map[signum])
            return False

    def unregister_signal(self, signum):
        """
        Unregister the given signal to the previous handler we have recorded.
        This also sets whether the signal has been caught or not to False.

        If the given signal is not registered, we return False.

        :type signum: int
        :param signum: The identifying integer value of the signal to check.
        (use signal.SIGINT, signal.SIGTERM, etc.)

        return: True if we successfully unregistered a signal, or False if the
        the given signal was not registered in the first place.

        """
        with self.__h_lock:
            if signum in self.__prev_handlers:
                self._log.debug("Restoring previous handler for %s",
                                self._sig_map[signum])
                signal.signal(signum, self.__prev_handlers[signum])
                del self.__prev_handlers[signum]
                # unregister signal caught value
                self.reset_signal(signum)
                return True
            self._log.debug("%s never registered.", self._sig_map[signum])
            return False

    def signals_registered(self):
        """
        Return what signals are currently registered in this handler

        :return: what signals are currently registered in this handler
        :rtype: list of int

        """
        return self.__prev_handlers.keys()

    def reset_signal(self, signum):
        """
        Reset our knowledge of whether a particular signal has been caught.
        This does NOT unregister any handlers.

        If the given signal is not registered, we return False.

        :type signum: int
        :param signum: The identifying integer value of the signal to check.
        (use signal.SIGINT, signal.SIGTERM, etc.)

        :return: True if the signal catch record was reset, and False if the
        given signal is not registered as being monitored.

        """
        with self.__s_lock:
            if signum in self.__signal_caught:
                self._log.debug("Resetting %s catch boolean",
                                self._sig_map[signum])
                del self.__signal_caught[signum]
                return True
            self._log.debug("%s never caught, nothing to reset",
                            self._sig_map[signum])
            return False

    def reset(self):
        """
        Reset our knowledge of what signals have been caught. This does NOT
        unregister any handlers.

        """
        with self.__s_lock:
            self._log.debug("Resetting all signal catches")
            self.__signal_caught = {}

    def is_signal_caught(self, signum):
        """
        Check if we have caught the given signal since the creation of the
        object or since the last reset. We also return false if the given
        signal isn't being monitored.

        :type signum: int
        :param signum: The identifying integer value of the signal to check.
        (use signal.SIGINT, signal.SIGTERM, etc.)

        :rtype: bool
        :return; True if the signal has been seen since creation/reset, or
            False otherwise.

        """
        return self.__signal_caught.get(signum, False)
