"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import cPickle
import logging
import multiprocessing
import pymongo
import uuid


def register_mdb_loc(host, port):
    global _sa_mdb_host, _sa_mdb_port
    _sa_mdb_host = str(host)
    _sa_mdb_port = int(port)


def _get_global_mdb_client():
    global _sa_mdb_client, _sa_mdb_host, _sa_mdb_port
    try:
        _sa_mdb_client
    except NameError:
        try:
            _sa_mdb_client = pymongo.MongoClient(host=_sa_mdb_host,
                                                 port=_sa_mdb_port)
        except NameError:
            raise RuntimeError("register_mdb_loc not called first!")
    return _sa_mdb_client


class SharedAttribute (object):
    """
    Descriptor for a variable within a ControllerProcess that may be changed in
    the forked process and be effectively reflected in the main thread. The
    value such attributes carry should be considered volatile and potentially
    changing while the process it is contained in is executing. Once the process
    is done executing, the attribute in the main thread will reflect the final
    value set in the process.

    CPAttributes must be defined on the class level and MUST be initialized in
    the constructor of the object by setting it to something. Since descriptors
    only work when they're defined on the class level, we need to initialize
    them to a value in the constructor in order to create an instance-specific
    hook (lock and queue) that sub-process copies will pick up.

    This is so that instance specific values are initialized before forking. If
    it is not initialized, the necessary structures will not exist in the
    containing object for reference as well as not being transported across
    processes.

    NOTE
    ----
    There is some magic happening behind the scenes in python that some
    how allows the "shared" object to be updated in place and have that
    change reflected on other threads. However, this isn't very safe as
    data can and will be lost when updating that structure very fast. The
    proper method of using this structure is to always perform an explicit
    ``a = b`` action on the object. This ensures stability, however incurs
    a non-trivial amount of overhead when trying to retrieve from a
    ``SharedAttribute`` and then setting back into it right away, also as an
    additional multiprocessing.Lock() should be used in such a case.

    If only a single thread is updating the structure, it is wiser to manager
    a thread-local version of the structure, only *setting* to the shared
    structure when the local version is updated.

    """

    # component access and labels
    _access_tmpl = "_{id:s}_{name:s}"
    _lock = 'l'
    _hash = 'h'
    _m_client = 'cl'
    _m_db = 'd'
    _m_collection = 'co'

    # DB Connections initialized upon initialization, not construction.
    DB_NAME = 'smqtk'  # Default, but may be modified to change behavior
    COLLECTION = "SharedAttributes"
    # Doc format:
    #   { uuid: <str_uuid>,
    #     obj: <int>,
    #     pickle: <str_pickle_dump> }

    def __init__(self, init_val=None):
        self._uid = uuid.uuid4()
        self._init_val = init_val
        self._log = logging.getLogger('.'.join((self.__module__,
                                                self.__class__.__name__)))

    def _gen_var_names(self):
        lv = self._access_tmpl.format(id=self._uid, name=self._lock)
        hv = self._access_tmpl.format(id=self._uid, name=self._hash)
        clv = self._access_tmpl.format(id=self._uid, name=self._m_client)
        dbv = self._access_tmpl.format(id=self._uid, name=self._m_db)
        cov = self._access_tmpl.format(id=self._uid, name=self._m_collection)
        return lv, hv, clv, dbv, cov

    def _components_exist(self, obj):
        """
        True if components exist for the given object. False if not.
        """
        lv, hv, clv, dbv, cov = self._gen_var_names()
        return all(hasattr(obj, v) for v in (lv, hv, clv, dbv, cov))

    def _init_components(self, obj):
        """
        Initialize and set components to the given object. This should happen on
        the main thread.
        """
        lv, hv, clv, dbv, cov = self._gen_var_names()
        h = hash((self._uid, id(obj)))
        m_client = _get_global_mdb_client()
        m_db = m_client[self.DB_NAME]
        m_coll = m_db[self.COLLECTION]
        m_coll.update({'_id': h},
                      {'$set': {'pickle': cPickle.dumps(self._init_val)}},
                      upsert=True)

        # Set things into the object
        setattr(obj, lv, multiprocessing.Lock())
        setattr(obj, hv, h)
        setattr(obj, clv, m_client)
        setattr(obj, dbv, m_db)
        setattr(obj, cov, m_coll)

    def _get_components(self, obj):
        """
        Return this object's shared attribute building blocks. Creates them if
        they don't exist yet. Returns lock and database connection components.

        """
        if not self._components_exist(obj):
            self._init_components(obj)
        lv, hv, clv, dbv, cov = self._gen_var_names()
        return (getattr(obj, lv), getattr(obj, hv),
                getattr(obj, clv), getattr(obj, dbv), getattr(obj, cov))

    # noinspection PyUnusedLocal
    def __get__(self, obj, klass=None):
        # Returns the descriptor object itself when requested from the class
        # level.
        if obj is None:
            return self

        l, h, client, db, collection = self._get_components(obj)

        with l:
            query_doc = {'_id': h}
            ret_doc = collection.find_one(query_doc)
            return cPickle.loads(str(ret_doc['pickle']))

    def __set__(self, obj, value):
        if obj is None:
            raise AttributeError("Setting a SharedAttribute value from the "
                                 "class-level. This is not allowed.")

        l, h, client, db, collection = self._get_components(obj)

        with l:
            query_doc = {'_id': h}
            update_doc = {'$set': {'pickle': cPickle.dumps(value)}}
            collection.update(query_doc, update_doc)

    def __delete__(self, obj):
        if obj is None:
            return
        self.close(obj)

    def close(self, obj):
        assert obj
        _, _, client, _, _ = self._get_components(obj)
        client.close()
