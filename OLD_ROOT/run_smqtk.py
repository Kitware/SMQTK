#!/usr/bin/env python
"""

High-level system run script for the SMQTK system. This script encapsulates and
manages all running components for a locally based instance.

We require that the system setup script has been sourced. We check for its
signature and exit with failure if not found.

Usage:

  ./run_smqtk.py [options]]

Processes started and managed:
    - mongo database
    - FeatureManager instance server
    - static server
    - WebApp server
    - celery worker(s)

"""
__author__ = "Paul Tunison"


import abc
import smqtk_config
import logging
import os
import signal
import subprocess
import time


class SignalInterceptor (object):
    """
    Object encapsulating ability to intercept a signal and execute an ordered
    stack of actions for the given signal.

    Actions added should be functions with the same signature as required by the
    signal.signal() function (handler argument).

    """

    class Action (object):

        def __init__(self, name, f):
            self.name = name
            self.func = f

        def __eq__(self, other):
            return isinstance(other, SignalInterceptor.Action) \
                and self.name == other.name \
                and self.func == other.func

        def __hash__(self):
            return hash((self.name, self.func))

    BASE_ACTION_NAME = "__SI_Base_Action__"

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

        # Mapping of the previous signal handler for a signal
        #
        # If there is something for a signal in this map, that signal is
        # considered registered. If not present, it is not registered.
        #
        #: :type: dict of (int, list)
        self.__registered = {}

        # Int-to-bool map describing if a particular signal has been caught
        #
        # If a signal isn't in this map, assume False.
        #
        #: :type: dict of (int, bool)
        self.__signals_caught = {}

        # Function stack to be performed for a given signal
        #
        # There should only be an entry in here for a signal in that signal
        # has been registered.
        #
        #: :type: dict of (int, list of SignalInterceptor.Action)
        self.__action_stack = {}

    def _handle_signal(self, signum, stack):
        """ Call actions for a signal

        :param signum: Signal being handled
        :type signum: int
        :param stack: Frame object
        :type stack: None or frame

        """
        self._log.debug("Signal %d intercepted. Executing actions.", signum)
        for i, action in enumerate(self.__action_stack[signum]):
            self._log.debug("Signal %d action %d -- %s",
                            signum, i+1, action.name)
            action.func(signum, stack)

    def _basic_catch(self, signum, stack):
        """ Catch signal and register that it was caught.

        :param signum: Signal being handled
        :type signum: int
        :param stack: Frame object
        :type stack: None or frame

        """
        self._log.debug('Registering signal %d as caught', signum)
        self.__signals_caught[signum] = True

    def registered_signals(self):
        """ Return a frozenset of signals that have been registered.

        :return: Set of signals registered (integer keys)
        :rtype: frozenset of int

        """
        return frozenset(self.__registered.keys())

    def register(self, signum):
        """ Instate our handler for a certain signal

        Adds the default catch method to the action stack

        :param signum: Signal to register
        :type signum: int

        """
        if signum not in self.__registered:
            self._log.debug("Registering signal %d", signum)
            prev_handle = signal.signal(signum, self._handle_signal)
            self.__registered[signum] = prev_handle
            # Initializing action stack with base catch
            self.__action_stack[signum] = \
                [SignalInterceptor.Action(self.BASE_ACTION_NAME,
                                          self._basic_catch)]
        else:
            self._log.debug("Signal %d already registered", signum)

    def unregister(self, signum):
        """ Reinstate the previous handle for the given signal

        Clears our internal state for the given signal (i.e. action stack and
        caught flag is removed).

        If the given signal has not been registered, a KeyError is thrown

        :param signum: Signal to unregister
        :type signum: int

        """
        with self.__registered[signum] as prev_handler:
            self._log.debug("RUnregistering signal %d", signum)
            signal.signal(signum, prev_handler)
            del self.__registered[signum]
            del self.__signals_caught[signum]
            del self.__action_stack[signum]

    def signal_caught(self, signum):
        return self.__signals_caught.get(signum, False)

    def signal_actions(self, signum):
        """ Return a tuple of the actions registered for a given signal.

        :raises KeyError: The given signal has not been registered.

        :param signum: Signal to get the current actions for.
        :type signum: int

        """
        return tuple(self.__action_stack[signum])

    def action_add(self, signum, action_name, func):
        """ Add an action to be performed when the given signal is intercepted

        Actions are added, and subsequently executed, sequentially.

        :raises ValueError: The given action name matches the internal base
            catch action name. A different name should be chosen.

        :param signum: Signal to add an action to
        :type signum: int
        :param func: Action to be performed. Signature must be the same as
            required for the handler function in the signal.signal() method.
        :type func: function

        """
        if action_name == self.BASE_ACTION_NAME:
            raise ValueError("The given action name matches the internal base "
                             "action name (%s). Please pick a different name."
                             % self.BASE_ACTION_NAME)

        action_stack = self.__action_stack.get(signum, [])
        action_stack.append(self.Action(action_name, func))
        self.__action_stack[signum] = action_stack

    def action_remove(self, signum, action_name):
        """ Remove an action from the signal's action list

        Removes the first action in the signals current action stack based on
        the given action name. Only removes one action at a time (starting from
        the top of the stack), even if the same action name is registered more
        than once.

        :raises KeyError: The given signal is not registered.
        :raises ValueError: Given name does not match any currently registered
            action.

        :param signum: Signal to remove the given action from
        :type signum:
        :param action_name: Name of the action to remove
        :type action_name: str

        """
        idx_for_removal = None
        for i, action in enumerate(self.__action_stack[signum]):
            if action.name == action_name:
                idx_for_removal = i
                break
        if idx_for_removal:
            popped = self.__action_stack[signum].pop(idx_for_removal)
            self._log.debug("Removed action \'%s\'", popped.name)
        else:
            raise ValueError("Given action name (%s) does not match any "
                             "registered actions for signal %d.",
                             action_name, signum)


class SmqtkProcess (object):
    """ Base class of SMQTK System process encapsulation
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.logfile = None
        #: :type: subprocess.Popen
        self.proc = None

    @abc.abstractmethod
    def logfile_name(self):
        """
        :return: Log file name
        :rtype: str
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def args(self):
        """
        :return: Argument list for the process
        :rtype: list of str
        """
        raise NotImplementedError()

    def run(self, log_dir):
        """
        Run the process
        """
        self.logfile = open(os.path.join(log_dir, self.logfile_name()), 'w')
        self.proc = subprocess.Popen(self.args(),
                                     stdout=self.logfile,
                                     stderr=self.logfile)
        # print "%s -> Here is where we start running" % self.__class__.__name__

    def cleanup(self, *args):
        """ Clean-up after this process
        """
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)

        if self.logfile:
            self.logfile.close()
            self.logfile = None


def main():
    ###
    # Setting up logging
    #
    log_format = '%(levelname)7s - %(asctime)s - %(name)s.%(funcName)s - ' \
                 '%(message)s'
    log_formatter = logging.Formatter(log_format)
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(log_stream_handler)
    logging.getLogger().setLevel(logging.INFO)

    log = logging.getLogger("run_smqtk")

    ###
    # checking SMQTK environment presence
    #
    smqtk_system_setup = bool(os.environ.get('SMQTK_SYSTEM_SETUP', 0))
    if not smqtk_system_setup:
        log.critical("SMQTK System not set-up. Please source the "
                     "appropriate set-up script.")
        exit(1)

    ###
    # Parsing arguments
    #
    import optparse

    parser = optparse.OptionParser()

    parser_g_required = optparse.OptionGroup(parser, "Required")
    parser_g_required.add_option('--dbpath', type=str,
                                 help="Path to the database data directory.")

    parser_g_optional = optparse.OptionGroup(parser, "Optional")
    parser_g_optional.add_option('--celery-workers', type=int, default=1,
                                 help="Number of celery workers to spawn. This "
                                      "must be at least 1.")
    parser_g_optional.add_option('-l', '--log-dir', type=str,
                                 default=smqtk_config.WORK_DIR,
                                 help="Optionally specify a different "
                                      "directory to place log files in.")
    parser_g_optional.add_option('-v', '--verbose', action='store_true',
                                 default=False,
                                 help="Display more messages.")

    parser.add_option_group(parser_g_required)
    parser.add_option_group(parser_g_optional)

    opts, args = parser.parse_args()

    # argument error checking
    if (opts.dbpath is None) or (not os.path.isdir(opts.dbpath)):
        log.critical("Missing '--dbpath' option. Please specify location of "
                     "database data directory.")
        exit(1)
    else:
        # normalize pathing
        opts.dbpath = os.path.abspath(os.path.expanduser(opts.dbpath))

    if opts.celery_workers < 1:
        log.critical("Number of celery workers must at least be one! Given %d.",
                     opts.celery_workers)
        exit(1)

    if opts.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # normalizing log dir path
    opts.log_dir = os.path.abspath(os.path.expanduser(opts.log_dir))

    # Reporting setup
    log.info("Run Parameters:")
    log.info("  dbpath         : %s", opts.dbpath)
    log.info("  celery workers : %d", opts.celery_workers)
    log.info("  log dir        : %s", opts.log_dir)
    log.info("  verbose        : %s", opts.verbose)

    ###
    # Process Management
    #
    def log_action(signum, _):
        log.info("Intercepted signal %d", signum)

    log.info("Instating SignalInterceptor")
    si = SignalInterceptor()
    si.register(signal.SIGINT)
    si.action_add(signal.SIGINT, 'catch log', log_action)

    # Process concrete classes
    log.info("Constructing process classes")

    class MongoDB_Process (SmqtkProcess):

        def logfile_name(self):
            return 'mongodb.log'

        def args(self):
            return ['mongod', '--dbpath', opts.dbpath]

    class FMS_Process (SmqtkProcess):

        def logfile_name(self):
            return 'fms.log'

        def args(self):
            return ['FeatureManagerServer', '-c',
                    os.path.join(smqtk_config.ETC_DIR, 'featuremanager.config')]

    class Celery_Process (SmqtkProcess):

        def __init__(self, worker_num):
            super(Celery_Process, self).__init__()
            self.num = worker_num

        def logfile_name(self):
            return 'celery_worker_%d.log' % self.num

        def args(self):
            return ['celery', 'worker', '-A', 'WebUI.video_process.celeryapp']

    class StaticWeb_Process (SmqtkProcess):

        def logfile_name(self):
            return 'static_server.log'

        def args(self):
            return ['run_static_server.py']

    class WebApp_Process (SmqtkProcess):

        def logfile_name(self):
            return "webapp_server.log"

        def args(self):
            return ['run_WebApp.py']

    # Create and start processes
    log.info("Constructing process list")
    # noinspection PyListCreation
    #: :type: list of (str, SmqtkProcess)
    smqtk_processes = []
    smqtk_processes.append(('mongod', MongoDB_Process()))
    smqtk_processes.append(('fms', FMS_Process()))
    for n in range(opts.celery_workers):
        smqtk_processes.append(('celery_%d' % n, Celery_Process(n)))
    smqtk_processes.append(('static_server', StaticWeb_Process()))
    smqtk_processes.append(('webapp_server', WebApp_Process()))

    log.info("Starting processes")
    for name, proc in smqtk_processes:
        log.info("-> %s", name)
        proc.run(opts.log_dir)

    log.info("Registering clean-up in reverse order")
    for name, proc in reversed(smqtk_processes):
        si.action_add(signal.SIGINT, '%s - CleanUp' % name, proc.cleanup)

    log.info("Waiting for terminations (Ctrl-C)")
    while not si.signal_caught(signal.SIGINT):
        time.sleep(0.1)

        # Check for pre-mature process shutdown
        do_shutdown = False
        for name, proc in smqtk_processes:
            if proc.proc.poll() is not None:
                log.error("Detected process pre-mature shutdown: %s", name)
                do_shutdown = True
        if do_shutdown:
            for name, proc in smqtk_processes:
                proc.cleanup()
            break

    log.info("Waiting for subprocess completion...")
    for name, proc in reversed(smqtk_processes):
        proc.proc.wait()
        log.info("%s exited", name)

    log.info("Exiting")


if __name__ == '__main__':
    main()
