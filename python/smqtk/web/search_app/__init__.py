"""
Top level flask application
"""

import base64
import json
import os
import os.path
import threading
from typing import Dict

import flask
from flask_cors import cross_origin  # type: ignore
import six
from werkzeug.exceptions import NotFound
from werkzeug.wsgi import peek_path_info, pop_path_info

from smqtk.utils import DatabaseInfo
from smqtk.utils.dict import merge_dict
from smqtk.utils.mongo_sessions import MongoSessionInterface
from smqtk.web import SmqtkWebApp

from .modules.login import LoginMod
from .modules.iqr import IqrSearch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_csrf_token():
    """
    Create a random string token for CSRF protection.

    Uses ``os.urandom`` so this should be sufficient for cryptographic uses.

    """
    BYTES = 64
    return base64.b64encode(os.urandom(BYTES)).decode('utf-8')


class IqrSearchDispatcher (SmqtkWebApp):
    """
    Application that dispatches to IQR application instances per sub-path.  We
    can be seeded with a set amount of instances with the ``iqr_tabs``
    configuration section, which consists of a prefix key (what would be used in
    the URL) to the configuration for that instance.  A ``__default__`` is
    provided upon configuration generation to act as a template.  The
    ``__default__`` value is ignored at runtime.

    New IQR instances can be dynamically added via a POST to the url root
    (``/``).
    """

    # Prefixes that ignore dispatch to IQR application instances
    PREFIX_BLACKLIST = {
        'static',
        'login',
        'login.passwd',
        'logout',
    }

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(IqrSearchDispatcher, cls).get_default_config()
        merge_dict(c, {
            "mongo": {
                "server": "127.0.0.1:27017",
                "database": "smqtk",
            },
            # Each entry in this mapping generates a new tab in the GUI
            "iqr_tabs": {
                "__default__": IqrSearch.get_default_config(),
            },
        })
        return c

    def __init__(self, json_config):
        super(IqrSearchDispatcher, self).__init__(json_config)

        #
        # Database setup using Mongo
        #
        h, p = self.json_config['mongo']['server'].split(':')
        n = self.json_config['mongo']['database']
        self.db_info = DatabaseInfo(h, p, n)

        # Use mongo for session storage.
        # -> This allows session modification during Flask methods called from
        #    AJAX routines (default Flask sessions do not)
        self.session_interface = MongoSessionInterface(self.db_info.host,
                                                       self.db_info.port,
                                                       self.db_info.name)

        #
        # Misc. Setup
        #

        # Add 'do' statement usage
        self.jinja_env.add_extension('jinja2.ext.do')

        #
        # Modules
        #
        # Load up required and optional module blueprints
        #

        # Mapping of IqrSearch application instances from their ID string
        self.instances = {}  # type: Dict[str, IqrSearch]
        self.instances_lock = threading.Lock()

        # Login module
        self._log.info("Initializing Login Blueprint")
        self.module_login = LoginMod('login', self)
        self.register_blueprint(self.module_login)

        # IQR modules
        # - for each entry in 'iqr_tabs', initialize a separate IqrSearch
        #   instance.
        for tab_name, tab_config in six.iteritems(self.json_config['iqr_tabs']):
            if tab_name == "__default__":
                # skipping default config sample
                continue
            self._log.info("Initializing IQR instance '%s'", tab_name)
            self.init_iqr_app(tab_config, tab_name)

        #
        # Basic routing
        #

        @self.route('/', methods=['GET'])
        def index():
            # self._log.info("Session: %s", flask.session.items())
            # noinspection PyUnresolvedReferences
            return flask.render_template(
                "index.html", instance_keys=list(self.instances.keys()),
                debug=self.debug
            )

        @self.route('/', methods=['POST'])
        @cross_origin(origins='*', vary_header=True)
        @self.module_login.login_required
        def add_instance():
            """
            Initialize new IQR instance given an ID for that instance, and the
            configuration for it.
            """
            # TODO: Something where only user that created instance can access
            #       it?
            prefix = flask.request.form['prefix']
            config = json.loads(flask.request.form['config'])

            # the URL prefix of the new IqrSearch instance
            new_url = flask.request.host_url + prefix
            self._log.info("New URL with route: %s", new_url)

            self.init_iqr_app(config, prefix)

            return flask.jsonify({
                'prefix': prefix,
                'url': new_url,
            })

    @staticmethod
    def _apply_csrf_protect(app):
        # Establish CSRF protection

        # Synchronized keys also defined in ``/static/js/smqtk.vars.js``.
        # - Session key is also what's used in forms
        CSRF_FORM_KEY = CSRF_SESSION_TOKEN_KEY = "_csrf_token"
        CSRF_HEADER_KEY = "X-Csrf-Token"

        # CSRF Protection
        @app.before_request
        def csrf_protect():
            if flask.request.method in ["POST", "PUT", "DELETE"]:
                session_token = flask.session.get(CSRF_SESSION_TOKEN_KEY, None)

                # CSRF token will either be in header or form, but must be
                # present in one of them.
                req_header_token = flask.request.headers.get(CSRF_HEADER_KEY,
                                                             None)
                req_form_token = flask.request.form.get(CSRF_FORM_KEY, None)

                if not session_token or session_token != (req_header_token or
                                                          req_form_token):
                    flask.abort(400)

        def get_csrf_session_token():
            """
            Initialize and return a specific, secure token for CSRF mitigation
            per session.
            """
            # Initialize CSRF protection token in session if not there.
            if CSRF_SESSION_TOKEN_KEY not in flask.session:
                flask.session[CSRF_SESSION_TOKEN_KEY] = generate_csrf_token()
            return flask.session['_csrf_token']

        # Add CSRF templating helper token generator
        app.jinja_env.globals['csrf_form_key'] = CSRF_FORM_KEY
        app.jinja_env.globals['csrf_token_gen'] = get_csrf_session_token

        return app

    def init_iqr_app(self, config, prefix):
        """
        Initialize IQR sub-application given a configuration for it and a prefix

        :param config: IqrSearch plugin configuration dictionary
        :type config: dict

        :param prefix: URL prefix for the instance
        :type prefix: str

        :return: Application instance.
        :rtype: IqrSearch

        """
        with self.instances_lock:
            if prefix not in self.instances:
                self._log.info("Initializing IQR instance '%s'", prefix)
                self._log.debug("IQR tab config:\n%s", config)
                # Strip any keys that are not expected by IqrSearch
                # constructor
                expected_keys = list(IqrSearch.get_default_config().keys())
                for k in set(config).difference(expected_keys):
                    self._log.debug("Removing unexpected key: %s", k)
                    del config[k]
                self._log.debug("Base app config: %s", self.config)

                a = IqrSearch.from_config(config, self)
                a.config.update(self.config)
                a.secret_key = self.secret_key
                a.session_interface = self.session_interface
                a.jinja_env.add_extension('jinja2.ext.do')
                self._apply_csrf_protect(a)

                self.instances[prefix] = a
            else:
                self._log.debug("Existing IQR instance for prefix: '%s'",
                                prefix)
                a = self.instances[prefix]

        return a

    def get_application(self, prefix):
        """
        Get the application for the given ``prefix`` or the NotFound exception
        if an application does not yet exist for the ``prefix``.

        :param prefix: Prefix name of the IQR application instance
        :type prefix: str

        :return: Application instance or None if there is no instance for the
            given ``prefix``.
        :rtype: IqrSearch | None

        """
        with self.instances_lock:
            return self.instances.get(prefix, None)

    def __call__(self, environ, start_response):
        path_prefix = peek_path_info(environ)
        self._log.debug("Base application __call__ path prefix: '%s'",
                        path_prefix)

        if path_prefix and path_prefix not in self.PREFIX_BLACKLIST:
            app = self.get_application(path_prefix)
            if app is not None:
                pop_path_info(environ)
            else:
                self._log.debug("No IQR application registered for prefix: "
                                "'%s'", path_prefix)
                app = NotFound()
        else:
            self._log.debug("No prefix or prefix in blacklist. "
                            "Using dispatcher app.")
            app = self.wsgi_app

        return app(environ, start_response)

    def run(self, host=None, port=None, debug=False, **options):
        # Establish CSRF protection
        self._apply_csrf_protect(self)

        super(IqrSearchDispatcher, self).run(host, port, debug, **options)


APPLICATION_CLASS = IqrSearchDispatcher
