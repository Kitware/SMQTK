"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging


class ECDBackendInterface (object):
    """
    Abstract base class for creating a backbone for use in a ECDStore,

    Implementation classes must override the following methods:
        - ``store(...)``
        - ``get(...)``
        - ``get_by(...)``

    More information about each method can be found in the doc-string in the
    interface method definitions. No super call should be made as these methods
    will do nothing.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        """
        Interface class for creating a backbone for use in a ECDStore object.

        Initializes a logging object, ``_log``, extending the namespace of the
        module with the name of the implementation class's name.

        """
        self._log = logging.getLogger('.'.join([__name__,
                                                self.__class__.__name__]))

    @abc.abstractmethod
    def store(self, elements, overwrite=False):
        """
        Store one or more ECDStoreElement objects.

        :param elements: single or iterable of elements to store
        :type elements: ECDStoreElement or Iterable of ECDStoreElement
        :param overwrite: If an element already exists at the provided key
            location, overwrite that stored element's value with the value in
            the given element, preventing an exception from being thrown.
        :type overwrite: bool

        :raises ECDDuplicateElementError: Tried to insert a duplicate element.

        """
        return

    @abc.abstractmethod
    def get(self, model_id, clip_id):
        """
        Query for a single ECDStoreElement that exactly matched the provided
        keys. There may be no elements for the given keys, raising and
        exception.

        :param model_id: The model id to query for
        :type model_id: str
        :param clip_id: The clip id to query for.
        :type clip_id: int

        :raises ECDNoElementError: No elements in storage that match the given
            keys.

        :return: The ECDStoreElement for the given keys.
        :rtype: ECDStoreElement

        """
        return

    @abc.abstractmethod
    def get_by(self, model_id=None, clip_id=None):
        """
        Retrieve a tuple of ECDStoreElement objects that all, or in part, match
        the specified keys. If no keys are specified, we will return all
        elements stored.

        :param model_id: The model id of the element
        :type model_id: str or None
        :param clip_id: The clip id of the element
        :type clip_id; int or None
        :return: A tuple of matching ECDStoreElement objects that match the
            query. This may be an empty tuple if there were no matching results.
        :rtype: tuple of ECDStoreElements

        """
        return