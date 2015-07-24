"""
LICENSE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
"""

import abc
import logging
import os

class Quantization (object):
    '''
    This class is the base class for quantizing some descriptor matrices. The actual
    implementation (k-means, etc.) is up to the user to implement in a child class. This rather defines
    the interface which a Quantization object should have.

    :param work_dir: Directory where the temporary files/working files should be stored
        during quantization. Relative paths are treated relative to ``smqtk_config.DATA_DIR``.
    :type work_dir: str | unicode

    :param quantization_filepath: Directory in which to write the generated codebook. Relative paths are
        treated relative to ``smqtk_config.DATA_DIR``.
    :type quantization_filepath: str | unicode

    :param label: Label of this quantization (can correspond to the ContentDescriptor if you'd like)
        but is general.
    :type label: str | unicode"
    '''
    def __init__(self,
        work_dir, 
        quantization_filepath, 
        label, 
        **kwargs):
        self._work_dir = work_dir
        self._quantization_filepath = quantization_filepath
        self._label = label

        self._log = logging.getLogger('.'.join([Quantization.__module__,
                                            Quantization.__name__]))

        self._quantization = None

        return

    @classmethod
    def is_usable(cls):
        # TODO -- are there general usability constraints?
        return True

    @property
    def quantization_filepath(self):
        '''
        Where we store the quantization.
        '''
        return self._quantization_filepath

    @abc.abstractmethod
    def generate_quantization(self):
        pass

    @property
    def quantization(self):
        return self._quantization

    @property
    def _descriptor_matrices(self):
        return self._descriptor_matrices

def get_quantizers():
    """
    Discover and return ContentDescriptor classes found in the given plugin
    search directory. Keys in the returned map are the names of the discovered
    classes, and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``CONTENT_DESCRIPTOR_CLASS``, which can either be a single class object or
    an iterable of class objects, to be exported. If the variable is set to
    None, we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same name and casing as the module. If
    neither are found, the module is skipped.

    :return: Map of discovered class object of type ``ContentDescriptor`` whose
        keys are the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "QUANTIZER_CLASS"
    return get_plugins(__name__, this_dir, helper_var, Quantization,
                       lambda c: c.is_usable())