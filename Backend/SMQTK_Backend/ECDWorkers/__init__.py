"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


ECD Workers
-----------
The top-level area of this module contains the base definition of classifier
processes (including fusion process as it is considered a really high level
classifier). All classifier implementations should subclass this class type.
Sub-classes must implement the ``_process`` method of the class as well as any
other utility methods that it might need. This method will be called with work
inputs and is expected to output ECDStoreElement objects for storage.

The name assigned to the process should directly correlate to the names model ID
of store entries created.


Classifiers
-----------
Classifier implementations must live in a sub-module within the ``classifiers``
sub-module. Submodules that will be found by the system must start with a
letter, lower or upper case, and consist of alphanumeric characters (including
underscores). Modules that are comprised differently than this will be ignored.
The name of the module will become the classifier type's label. Within that
module's __init__.py file, two variables must be defined:
    - ``learn``
        - must be an ECDClassifierBase implementation that deals with learning,
          writing models to the given model file path and using the params file
          parameter for configuration.
    - ``search``
        - must be set to an ECDClassifierBase implementation that deals with
          searching, which loads up the provided model path for use, ignoring
          the params file parameter.

This implementation hierarchy then allows for a single structure to be imported
here to get access to the various classes through relatively simple means.


Fusion
------
Fusion process implementations should be placed in the ``fusion`` sub-module.
For now, specific fusion implementations should be imported, but it may be the
case in the future that a plug-in like thing is used (as with the classifiers)
so that implementation use may be chosen at configuration time (or even run
time). Implementations of fusion processes should inherit from the
``BaseFusion`` class in the ``fusion`` sub-module, which acts very similar to
the worker base class (different initialization / runtime).

Fusion processes know what workers to send sub-work messages to based on a
classifier configuration. This configuration object (dictionary) is assigned to
a fusion process upon construction within the ECDController. For formatting
information, see the ECDController documentation.

"""

import abc
import logging
import multiprocessing
import os.path as osp

from ..ControllerProcess import ControllerProcess
from ..ECDQueuePacket import ECDQueuePacket
from ..ECDStore.errors import ECDDuplicateElementError
from ..utils import SignalHandler


class ECDWorkerBaseProcess (ControllerProcess):
    """
    Abstract definition for base classifiers. For each base classifier type,
    there should be a learning and search implementation. Constructors should
    not deviate from the given skeleton so as to remain abstractly callable.

    Work is input into the system via ECDQueuePacket objects along the
    work_queue. A None value along the queue, or a packet with a clip ID valid
    of None, tells the processes to shutdown.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, model_path, learning_params_file_path,
                 vcd_controller, ecd_controller,
                 storage_overwrite=False, skip_bad_packets=True):
        """
        Initialization of the classifier process performed on the main thread.

        :param name: Name for this base process. For classifier workers, this is
            their model ID.
        :type name: str
        :param model_path: Path to a model file. This should be absolute. If a
            relative path is given, it is interpreted as relative to the ECD
            Controller's work directory.
        :type model_path: str
        :param learning_params_file_path: Path to a learning parameters file.
            None if there is no learning parameters file. If a path is given,
            it should be absolute. Relative paths will be interpreted as
            relative to the ECD Controller's data directory.
        :type learning_params_file_path: str or None
        :param vcd_controller: The VCD Controller object so we can know about
            the VCD store and work queue.
        :type vcd_controller: SMQTK_Backend.VCDController.VCDController
        :param ecd_controller: The ECD Controller object so we can know about
            the ECD store and work queue.
        :type ecd_controller: SMQTK_Backend.ECDController.ECDController
        :param storage_overwrite: Allow this process to overwrite data in the
            configured ECD storage space, as found through the controller.
        :type storage_overwrite: bool
        :param skip_bad_packets: Warn about, but otherwise ignore bad objects
            found in the work queue.
        :type skip_bad_packets: bool

        """
        super(ECDWorkerBaseProcess, self).__init__(name)

        self._log = logging.getLogger('.'.join([__name__,
                                                self.__class__.__name__]))

        # Normalizing paths relative to the ECD Controller's work directory if
        # they are not already absolute.
        #noinspection PyProtectedMember
        self._model_file_path = osp.join(ecd_controller._work_dir, model_path) \
            if model_path else None
        #noinspection PyProtectedMember
        self._learning_params_file_path = learning_params_file_path \
            if learning_params_file_path is None \
            else osp.join(ecd_controller._data_dir, learning_params_file_path)

        self._vcdc_store = vcd_controller.store
        self._vcdc_queue = vcd_controller.work_queue

        self._ecdc = ecd_controller
        self._ecdc_store_overwrite = bool(storage_overwrite)

        self._skip_bad_packets = skip_bad_packets

        # For transmission of ECDQueuePacket objects
        self.work_queue = multiprocessing.Queue()

        self.sh = SignalHandler()

    def __repr__(self):
        return "ECDWorkerBaseProcess" \
               "{name: %s, uuid: %s, is_alive: %s," \
               " model_file: %s, learning_params: %s}" \
               % (self.name, self.uuid,self.is_alive(),
                  self._model_file_path, self._learning_params_file_path)

    def _run(self):
        """
        Main runtime of the classifier worker process.
        """
        # Need to override the signal handler inherited from the ECDC.
        # Want to raise a KeyboardInterrupt exception on the first instance of
        # it, but not afterwords so that we can shutdown without interruption
        # (except for signals that murder the process).
        #noinspection PyAttributeOutsideInit
        self._do_interrupt = False

        def interrupt_handle(signum, _):
            if not self._do_interrupt:
                # pretend we didn't catch it
                self.sh.reset_signal(signum)
                self._do_interrupt = True
                raise KeyboardInterrupt("Raised from signal handler")

        # Register for SIGINT
        self.sh.register_signal(2, interrupt_handle)

        try:
            self._log.debug("Entering ECDW initialize()")
            self._initialize()

            active = True
            while active:
                #: :type: ECDQueuePacket or None
                work_packet = self.work_queue.get()

                if not (work_packet is None
                        or isinstance(work_packet, ECDQueuePacket)):
                    if self._skip_bad_packets:
                        self._log.warn("Bad packet found ('%s'). Skipping and "
                                       "waiting for next packet.",
                                       str(work_packet))
                        continue
                    else:
                        msg = "Bad packet received: %s" % str(work_packet)
                        self._log.error(msg)
                        raise RuntimeError(msg)

                if work_packet is None or work_packet.clips is None:
                    # shutdown
                    active = False
                else:
                    # noinspection PyTypeChecker
                    # reason: This will not be None by this point.
                    elements = self._process(work_packet)
                    if elements:
                        # Determine store to use based on packet. If there is a
                        # search context, use the collection'
                        if work_packet.custom_collection:
                            coll = str(work_packet.custom_collection)
                            ecd_store = self._ecdc.store(coll)
                        else:
                            ecd_store = self._ecdc.store()

                        self._log.debug("Elements received: %s", elements)
                        try:
                            ecd_store.store(
                                elements, overwrite=self._ecdc_store_overwrite
                            )
                        except ECDDuplicateElementError:
                            self._log.warn("Attempted duplicate element "
                                           "insertion. Skipping output from "
                                           "this work packet.")

            # Nullify ability to throw KeyboardInterrupt with SIGINT
        except KeyboardInterrupt:
            self._log.info("ECDWorker '%s' caught keyboard interruption. "
                           "Exiting after shutdown.", self.name)

        finally:
            self._log.debug("Entering ECDW shutdown()")
            self._shutdown()

    #noinspection PyMethodMayBeStatic
    def _initialize(self):
        """
        Initialize the state of the ECD worker on the process-side (different
        memory space from when object initialized).

        NOTE - This method may be interrupted.

        """
        return

    @abc.abstractmethod
    def _process(self, work_packet):
        """
        Perform work on the given ECDQueuePacket of data, returning
        ECDStoreElement objects for storage.

        :param work_packet: A data packet
        :type work_packet: ECDQueuePacket
        :return: ECDStoreElement objects containing the data to store.
        :rtype: ECDStoreElement

        """
        return

    #noinspection PyMethodMayBeStatic
    def _shutdown(self):
        """
        Safe shutdown procedures for this classifier implementation.

        NOTE - Cannot assume here that initialization or main processing was not
        interrupted.

        """
        return
