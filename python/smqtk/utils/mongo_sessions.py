"""
Mongo Session wrapper

Allows session updating within AJAX routines.

Taken from: http://flask.pocoo.org/snippets/110/

"""

from datetime import datetime, timedelta
import logging
from uuid import uuid4

from flask.sessions import SessionInterface, SessionMixin
from werkzeug.datastructures import CallbackDict
from pymongo import MongoClient


LOG = logging.getLogger(__name__)


class MongoSession(CallbackDict, SessionMixin):

    def __init__(self, initial=None, sid=None):
        super(MongoSession, self).__init__(initial, MongoSession.on_update)
        self.sid = sid
        self.modified = False

    def on_update(self):
        self.modified = True


class MongoSessionInterface(SessionInterface):

    def __init__(self, host='localhost', port=27017,
                 db='', collection='sessions', delete_on_empty=False):
        client = MongoClient(host, port)
        self.store = client[db][collection]
        self._delete_on_empty = delete_on_empty

    def open_session(self, app, request):
        sid = request.cookies.get(app.session_cookie_name)
        if sid:
            stored_session = self.store.find_one({'_id': sid})
            if stored_session:
                if stored_session.get('expiration') > datetime.utcnow():
                    LOG.debug("Returning existing MongoSession instance for "
                              "SID={}".format(sid))
                    return MongoSession(initial=stored_session['data'],
                                        sid=stored_session['_id'])
        sid = str(uuid4())
        LOG.debug("Returning NEW MongoSession instance for SID={}"
                  .format(sid))
        return MongoSession(sid=sid)

    def save_session(self, app, session, response):
        domain = self.get_cookie_domain(app)
        if self._delete_on_empty and not session:
            LOG.debug("Session cookie content was empty, deleting.")
            response.delete_cookie(app.session_cookie_name, domain=domain)
            # TODO: Delete from mongo as well?
            return
        # Update expiration due to access.
        expiration = self.get_expiration_time(app, session)
        if not expiration:
            expiration = datetime.utcnow() + timedelta(hours=1)
        self.store.update({'_id': session.sid},
                          {'data': session,
                           'expiration': expiration},
                          upsert=True)
        LOG.debug("Setting session cookie for SID={}".format(session.sid))
        response.set_cookie(app.session_cookie_name, session.sid,
                            expires=expiration,
                            httponly=True, domain=domain)
