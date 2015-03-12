# coding=utf-8
"""
Top level MASIR web application object
"""

import flask
import logging
import mimetypes
import os
import os.path as osp

from masir.MongoDatabaseInfo import MongoDatabaseInfo
from masir.web.MongoSessions import MongoSessionInterface


mimetypes.add_type('image/png', '.png')
mimetypes.add_type('video/ogg', '.obv')
mimetypes.add_type('video/webm', '.webm')

script_dir = osp.dirname(osp.abspath(__file__))


# noinspection PyUnusedLocal
class MasirApp (flask.Flask):

    ENV_CONFIG = 'MASIR_CONFIG'

    @property
    def _log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    def __init__(self, config_file_path=None):
        super(MasirApp, self).__init__(
            'masir',
            static_folder=osp.join(script_dir, 'static'),
            template_folder=osp.join(script_dir, 'templates')
        )

        #
        # Configuration setup
        #
        config_default_loaded = config_env_loaded = config_file_loaded = None

        # Load default -- This should always be present, aka base defaults
        self.config.from_object('masir_config')
        config_default_loaded = True

        # Load from env var if present
        if self.ENV_CONFIG in os.environ:
            self._log.info("Loading config from env var (%s)...",
                           self.ENV_CONFIG)
            self.config.from_envvar(self.ENV_CONFIG)
            config_env_loaded = True

        # Load from configuration file if given
        if config_file_path and osp.isfile(config_file_path):
            config_file_path = osp.expanduser(osp.abspath(config_file_path))
            self._log.info("Loading config from file (%s)...", config_file_path)
            self.config.from_pyfile(config_file_path)
            config_file_loaded = True

        self._log.debug("Config defaults loaded : %s", config_default_loaded)
        self._log.debug("Config from env loaded : %s", config_env_loaded)
        self._log.debug("Config from file loaded: %s", config_file_loaded)
        if not (config_default_loaded or config_env_loaded or config_file_loaded):
            raise RuntimeError("No configuration file specified for loading. "
                               "(%s=%s) (file=%s)"
                               % (self.ENV_CONFIG,
                                  os.environ.get(self.ENV_CONFIG, None),
                                  config_file_path))

        self._log.debug("Configuration loaded: %s", self.config)

        #
        # Setting the secret key
        #
        self.secret_key = self.config['SECRET_KEY']

        #
        # Database setup
        #
        host, port = self.config['MONGO_SERVER'].split(":")
        self.db_info = MongoDatabaseInfo(host, port, 'masir')

        # Session interface using MongoDB
        # -> This allows session modification during AJAX routines (default
        #    Flask sessions do not)
        self.session_interface = MongoSessionInterface(self.db_info.host,
                                                       self.db_info.port,
                                                       self.db_info.db)

        #
        # Misc. Setup
        #

        # Add 'do' statement usage
        self.jinja_env.add_extension('jinja2.ext.do')

        #
        # Basic routing
        #

        @self.route('/home')
        @self.route('/')
        def masir_index():
            self._log.info("Session: %s", flask.session.items())
            return flask.render_template("index.html")

        #
        # Modules
        #
        # Load up required and optional module blueprints
        #

        self._log.debug("Importing Login module")
        from .mods.login import LoginMod
        self.module_login = LoginMod(self)
        self.register_blueprint(self.module_login)

        self._log.debug("Importing Search module")
        from .mods.search import SearchMod
        self.module_search = SearchMod(self)
        self.register_blueprint(self.module_search,
                                url_prefix="/search")

    def run(self, host=None, port=None, debug=False, **options):
        """
        Override of the run method, drawing running host and port from
        configuration by default. 'host' and 'port' values specified as argument
        or keyword will override the app configuration.
        """
        super(MasirApp, self).run(host=(host or self.config['RUN_HOST']),
                                  port=(port or self.config['RUN_PORT']),
                                  **options)