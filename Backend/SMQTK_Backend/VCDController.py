"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os
import os.path as osp
import multiprocessing

from .ControllerProcess import ControllerProcess
from .utils import SafeConfigCommentParser
from .utils import SignalHandler
from .VCDStore import VCDStore


class VCDQueuePacket (object):
    """
    Data packet that should be sent along the VCDController queue
    """

    def __init__(self, clip_path, worker_id):
        """
        :param clip_path: The path to the clip that should be processed.
        :type clip_path: str
        :param worker_id: The string id tag of the worker process that should
            work on the given clip
        :type worker_id: str

        """
        self.clip_path = str(clip_path)
        self.worker_id = str(worker_id)

    @classmethod
    def from_packet(cls, packet):
        """
        Construct a new packet from a another packet.

        :param packet: The other packet to construct from.
        :type packet: VCDQueuePacket

        """
        return VCDQueuePacket(packet.clip_path, packet.worker_id)


class VCDController (ControllerProcess):
    """
    Process that manages a dispatch queue for processing requests, spawning
    workers based on known VCDWorkers implementations to produce data that is
    inserted into the registered VCDStore.

    """

    CONFIG_SECT = 'vcd_controller'

    @classmethod
    def generate_config(cls, config=None):
        """
        Generate and return the configuration for this VCD Controller.

        :param config: And existing configuration object to add to. By default
            we will create a new config object and return it. Else the provided
            config object is modified and returned.
        :type config: SafeConfigCommentParser or None
        :return: A new config object, of the same one provided, with new
            sections/options for this controller.
        :rtype: SafeConfigCommentParser

        """
        if config is None:
            config = SafeConfigCommentParser()
        sect = cls.CONFIG_SECT
        if sect not in config.sections():
            config.add_section(sect,
                               'Options for the VCD process controller')

        config.set(sect, 'store_name', osp.join(os.getcwd(), 'vcd_store'),
                   'The name of the VCD storage container.\n'
                   '\n'
                   'For the default VCDStore implementation, a database '
                   'file will need to be created or loaded. For that case, '
                   'this "name" is treated as a file path. If a preexisting '
                   'SQLite3 databate already exists from a previous run/'
                   'installation, a relative path may be given that will be '
                   'interpreted relative to the working directory, or an '
                   'absolute path may be given.')

        return config

    def __init__(self, config):
        """
        :param config: The configuration object for this controller and
            sub-implementations.

        """
        super(VCDController, self).__init__('VCDController')

        ### Helpers
        cget = lambda k: config.get(sect, k)

        sect = self.CONFIG_SECT
        self._vcd_store = VCDStore(fs_db_path=cget('store_name'))

        self.work_queue = multiprocessing.Queue()

    @property
    def store(self):
        """
        @return: the VDC storage object.
        @rtype: SMQTK_Backend.VCDStore.VCDStore

        """
        return self._vcd_store

    def queue(self, packet):
        """
        Submit a job message to this controller. A None value given will
        shutdown the controller after current processing completes.

        :param packet: The data packet to queue up.
        :type packet: VCDQueuePacket or None

        """
        assert packet is None or isinstance(packet, VCDQueuePacket), \
            "Not given a VCDQueuePacket to transport!"
        if packet is None:
            self._log.info("Inserted None packet into VCDController. Closing "
                           "runtime.")
        try:
            self.work_queue.put(packet)
        except AssertionError:
            self._log.warning("Failed to insert into a closed work queue! "
                              "Controller must have already been shutdown.")

    def _run(self):
        """
        Runtime loop that waits for work elements to come along. When work
        elements arrive, spawn off worker processes into the pool to run.

        The runtime is interrupted when given a shutdown signal along the queue.
        This shutdown signal is a simple None object.

        Normal messages are expected to be of the format:
            "clip_path"

        """
        # process_pool = multiprocessing.Pool()

        # TODO: The runtime that launches the processes upon retrieval of
        #       messages

        signal_handler = SignalHandler()
        # signal_handler.register_signal(2)  # SIGINT

        # TODO: Stuff

        # signal_handler.unregister_signal(2)  # SIGINT

        return

    def join(self, timeout=None):
        # run method will exit after receiving the termination signal
        self.work_queue.close()
        self.work_queue.join_thread()
        super(VCDController, self).join(timeout)
