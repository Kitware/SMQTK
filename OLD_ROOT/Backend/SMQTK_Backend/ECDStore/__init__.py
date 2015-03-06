"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

# pass-through import
from .ECDStoreElement import ECDStoreElement

from .implementations import ECDBackendInterface, MongoDB_ECDBackend


class ECDStore (object):
    """
    Abstract storage of ECD computation results as produced by ECD workers.

    Contents are stored i ECDStoreElement objects which encapsulate a
    probability and its associated metadata (model and clip id).

    """

    def __init__(self, *args, **kwargs):
        """
        Abstract storage of ECD generated clip probabilities.

        Contents are stored in a key-value schema, where the keys are the
        model id and the clip id.

        By default, a MongoDB backend is used, but a different one may be
        specified through the ``_backend`` keyword argument, which should be
        passed the class of the backend to use (from the implementations
        directory in this module). The MongoDB backend expects 3 arguments at
        construction:
            - ``host`` (DEFAULT: "localhost")
            - ``port`` (DEFAULT: 27017)
            - ``database`` (DEFAULT: "SMQTK_ECD_STORE")
            - ``collection`` (DEFAULT: "ecd_store")

        """
        if '_backend' in kwargs:
            backend_type = kwargs.get('_backend')
            assert issubclass(backend_type, ECDBackendInterface), \
                "Provided backend was not a subclass of the standard interface!"
            kwargs.pop('_backend')
            self._backend = backend_type(*args, **kwargs)
        else:
            self._backend = MongoDB_ECDBackend(*args, **kwargs)

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
        self._backend.store(elements, overwrite)

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
        return self._backend.get(model_id, clip_id)

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
        return self._backend.get_by(model_id, clip_id)
