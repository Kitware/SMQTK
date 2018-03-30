import os
import abc
import logging
from smqtk.utils import plugin

try:
    import torch
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
else:
    from torch import nn

__author__ = 'bo.dong@kitware.com'

class PyTorchModelElement(plugin.Pluggable):
    """
    Parent class for all PyTorch model interfaces.
    """

    @abc.abstractmethod
    def model_def(self):
        """
        Internal method that return PyTorch Model definition.
        This returns a torch.nn.Module.

        :return: PyTorch model definition
        :rtype: torch.nn.Module class

        """

    @abc.abstractmethod
    def transforms(self):
        """
        Internal method that return transforms used for
        preprocessing the inputs.
        This returns a torchvision.transfroms.

        :return: transforms for preprocessing
        :rtype: torchvision.transforms

        """


def get_pytorchmodel_element_impls(reload_modules=False):
    """
    Discover and return discovered ``PyTorchModelElement`` classes. Keys in the
    returned map are the names of the discovered classes, and the paired values
    are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that begin
          with an alphanumeric character),
        - python modules listed in the environment variable ``PYTORCHMODEL_ELEMENT_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``PYTORCHMODEL_ELEMENT_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``PyTorchModelElement``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "PYTORCHMODEL_ELEMENT_PATH"
    helper_var = "PYTORCHMODEL_ELEMENT_CLASS"
    return plugin.get_plugins(__name__, this_dir, env_var, helper_var,
                              PyTorchModelElement, reload_modules)