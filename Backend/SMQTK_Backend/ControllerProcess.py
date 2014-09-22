"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import multiprocessing
import uuid


class ControllerProcessInfo (object):
    """
    Due to process objects not being very transportable, this object
    encapsulates the metadata surrounding the process.
    """

    def __init__(self, name, uuid, is_alive, exitcode):
        self.name = name
        self.uuid = uuid
        self.is_alive = is_alive
        self.exitcode = exitcode

    def __repr__(self):
        return "ControllerProcessInfo" \
               "{name: %s, uuid: %s, is_alive: %s}" \
               % (self.name, self.uuid, self.is_alive)


class ControllerProcess (multiprocessing.Process):
    """
    A process that is run and controlled by the SMQTK Controller. It may have
    dependencies on other ControllerProcesses

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        """
        Create a new controller process. Each process can only be "started"
        once.

        Being an abstract base class, this controller process cannot be
        instantiated directly. Subclasses must provide an implementation to this
        class's abstract methods.

        :param name: A name for this process.
        :type name: str

        :raise TypeError: Not all abstract methods or properties have been
            overridden.
        :raise AssertionError: Subclass didn't provide a value for this
            process's NAME.

        """
        # no target since we override the run method
        super(ControllerProcess, self).__init__(None, None, name, (), {})

        self._log = logging.getLogger('.'.join((self.__module__,
                                                self.__class__.__name__)))
        self._uuid = uuid.uuid4()

    def __repr__(self):
        return "ControllerProcess" \
               "{name: %s, uuid: %s, is_alive: %s}" \
               % (self.name, self.uuid, self.is_alive())

    @property
    def uuid(self):
        """
        :return: Unique identifier of this process instance.
        :rtype: UUID
        """
        return self._uuid

    def get_info(self):
        """
        Return the info metadata packet for this process
        """
        return ControllerProcessInfo(self.name, self.uuid, self.is_alive(),
                                     self.exitcode)

    def run(self):
        """
        Run the abstractly defined _run method (to be implemented by
        sub-classes) with output capturing.

        If a sub-class's ``_run`` method is designed to return a value, the
        default implementation of multiprocess.Process would not capture it.
        Now, the value is captured in the parameter and retrievable through
        the ``run_output`` accessor.

        Values that is returned should be picklable (for transport back from
        the spawned processes).

        """
        # Placeholder structure for more advanced functionality.
        #try:
        #    self._run()
        #except Exception:
        #    raise
        self._run()

    @abc.abstractmethod
    def _run(self):
        """
        Self-registered method that will be executed when this process is
        started.

        """
        return
