"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import pymongo
import pymongo.errors

from . import ECDBackendInterface
from .. import ECDStoreElement
from .. import errors


class MongoDB_ECDBackend (ECDBackendInterface):
    """
    A MongoDB Database specific backend implementation of the ECDStore
    interface. This required some parameters to know what database to connect
    to. Very basic defaults are provided for them, but it is usually not
    desirable to use them.

    """

    def __init__(self, host='localhost', port=27017, database='SMQTK_ECD_STORE',
                 collection='ecd_store'):
        super(MongoDB_ECDBackend, self).__init__()

        # Database connection initialization
        self._mdb_client = pymongo.MongoClient(host, port)
        self._mdb_db = self._mdb_client[database]
        self._mdb_coll = self._mdb_db[collection]

        # mdb document keys
        self._dkey_model_id = 'model_id'
        self._dkey_clip_id = 'clip_id'
        self._dkey_prob = 'probability'

        # Creating indexes for the two key columns when querying based on just
        # those attributes.
        self._mdb_coll.create_index(self._dkey_model_id)
        self._mdb_coll.create_index(self._dkey_clip_id)

    def store(self, elements, overwrite=False):
        # Store one or more ECDStoreElements. Unless ``elements`` is an
        # ECDStoreElement object, assume its iterable
        if isinstance(elements, ECDStoreElement):
            #: :type: tuple of ECDStoreElement
            elements = (elements,)

        # Create documents from the elements
        docs = [None] * len(elements)
        for i, elem in enumerate(elements):
            docs[i] = {
                '_id': hash((elem.model_id, elem.clip_id)),
                self._dkey_model_id: elem.model_id,
                self._dkey_clip_id: elem.clip_id,
                self._dkey_prob: elem.probability
            }

        if overwrite:
            # Save entries (type of update) since we are providing an _id.
            # Can only do this one at a time due to API restrictions.
            for d in docs:
                self._mdb_coll.save(d)
        else:
            try:
                self._mdb_coll.insert(docs, continue_on_error=True)
            except pymongo.errors.DuplicateKeyError, ex:
                raise errors.ECDDuplicateElementError(
                    "Attempted insert of a duplicate element (error: %s)"
                    % str(ex)
                )

    def get(self, model_id, clip_id):
        ret = self._mdb_coll.find_one(hash((str(model_id), int(clip_id))))

        if ret is None:  # no element by the given keys
            raise errors.ECDNoElementError(
                "No ECDStoreElement for the given keys: "
                "{model_id=%s, clip_id=%d}"
                % (str(model_id), int(clip_id))
            )

        return ECDStoreElement(
            model_id=ret[self._dkey_model_id],
            clip_id=ret[self._dkey_clip_id],
            probability=ret[self._dkey_prob]
        )

    def get_by(self, model_id=None, clip_id=None):
        search_doc = {}
        if model_id is not None:
            search_doc[self._dkey_model_id] = str(model_id)
        if clip_id is not None:
            search_doc[self._dkey_clip_id] = int(clip_id)

        cursor = self._mdb_coll.find(search_doc)

        elems = []
        for doc in cursor:
            elems.append(
                ECDStoreElement(
                    model_id=doc[self._dkey_model_id],
                    clip_id=doc[self._dkey_clip_id],
                    probability=doc[self._dkey_prob]
                )
            )

        return tuple(elems)
