# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import os

from smqtk.utils import safe_create_dir


class Catalyst (object):
    """
    Abstract base class for encapsulating a method of fusing index ranking
    results.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, work_dir):
        self._data_dir = data_dir
        self._work_dir = work_dir

    @property
    def log(self):
        """
        :return: logging object for this class
        :rtype: logging.Logger
        """
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @property
    def data_directory(self):
        safe_create_dir(self._data_dir)
        return self._data_dir

    @property
    def work_directory(self):
        safe_create_dir(self._work_dir)
        return self._work_dir

    @abc.abstractmethod
    def fuse(self, *rank_maps):
        """
        Given one or more dictionaries mapping UID to probability value (value
        between [0,1]), fuse by some manner unto a single comprehensive mapping
        of UID to probability value.

        :raises ValueError: No ranking maps given.

        :param rank_maps: One or more rank dictionaries
        :type rank_maps: list of (dict of (int, float))

        :return: Fused ranking dictionary
        :rtype: dict of (int, float)

        """
        if not len(rank_maps):
            raise ValueError("No rankings provided.")

    @abc.abstractmethod
    def reset(self):
        """
        Reset this catalyst instance to its original state.
        """
        pass


def get_catalysts():
    """
    Discover and return Catalyst classes found in the given plugin search
    directory. Keys in the returned map are the names of the discovered classes,
    and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with an alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module we first look for a helper variable by the name
    ``CATALYST_CLASS``, which can either be a single class object or an iterable
    of class objects, to be exported. If the variable is set to None, we skip
    that module and do not import anything. If the variable is not present, we
    look for a class by the same name and casing as the module. If neither are
    found, the module is skipped.

    :return: Map of discovered class object of type ``Catalyst`` whose keys are
        the string names of the classes.
    :rtype: dict of (str, type)

    """
    from smqtk.utils.plugin import get_plugins
    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "CATALYST_CLASS"
    return get_plugins(__name__, this_dir, helper_var, Catalyst)
