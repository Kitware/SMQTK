"""
Top level flask application
"""

import flask
import json
import logging
import os.path

from SMQTK.FeatureDescriptors import get_descriptors
from SMQTK.Indexers import get_indexers

from SMQTK.utils.MongoSessions import MongoSessionInterface
from SMQTK.utils import DatabaseInfo, DataIngest, VideoIngest


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SMQTKSearchApp (flask.Flask):

    ENV_CONFIG = "SMQTK_SEARCHAPP_CONFIG"

    @property
    def log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    def __init__(self, config_filepath=None):
        super(SMQTKSearchApp, self).__init__(
            self.__class__.__name__,
            static_folder=os.path.join(SCRIPT_DIR, 'static'),
            template_folder=os.path.join(SCRIPT_DIR, 'templates')
        )

        #
        # Configuration setup
        #
        config_env_loaded = config_file_loaded = None

        # Load default -- This should always be present, aka base defaults
        self.config.from_object('smqtk_config')
        config_default_loaded = True

        # Load from env var if present
        if self.ENV_CONFIG in os.environ:
            self.log.info("Loading config from env var (%s)...",
                          self.ENV_CONFIG)
            self.config.from_envvar(self.ENV_CONFIG)
            config_env_loaded = True

        # Load from configuration file if given
        if config_filepath and os.path.isfile(config_filepath):
            config_file_path = os.path.expanduser(os.path.abspath(config_filepath))
            self.log.info("Loading config from file (%s)...", config_file_path)
            self.config.from_pyfile(config_file_path)
            config_file_loaded = True

        self.log.debug("Config defaults loaded : %s", config_default_loaded)
        self.log.debug("Config from env loaded : %s", config_env_loaded)
        self.log.debug("Config from file loaded: %s", config_file_loaded)
        if not (config_default_loaded or config_env_loaded or config_file_loaded):
            raise RuntimeError("No configuration file specified for loading. "
                               "(%s=%s) (file=%s)"
                               % (self.ENV_CONFIG,
                                  os.environ.get(self.ENV_CONFIG, None),
                                  config_filepath))

        self.log.debug("Configuration loaded: %s", self.config)

        #
        # Security
        #
        self.secret_key = self.config['SECRET_KEY']

        #
        # Database setup using Mongo
        #
        h, p = self.config['MONGO_SERVER'].split(':')
        n = "SMQTKSearchApp"
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

        # Login module
        self.log.info("Initializing Login Blueprint")
        from SMQTK.Web.common_flask_blueprints.login import LoginMod
        self.module_login = LoginMod('login', self)
        self.register_blueprint(self.module_login)

        # IQR modules
        from .modules.IQR import IQRSearch
        # TODO: At the moment, for simplicity, we're fixing the feature detector
        #       and indexer types. In the future this should either be moved
        #       to something that can be chosen by the user or a
        #       multi-feature/indexer fusion system.
        self.log.info("Loading Image Ingest")
        ingest_image = \
            DataIngest(os.path.join(self.config['DATA_DIR'],
                                    self.config['SYSTEM_CONFIG']['Ingest']['Image']),
                       os.path.join(self.config['WORK_DIR'],
                                    self.config['SYSTEM_CONFIG']['Ingest']['Image']))
        self.log.info("Loading Video Ingest")
        ingest_video = \
            VideoIngest(os.path.join(self.config['DATA_DIR'],
                                     self.config['SYSTEM_CONFIG']['Ingest']['Video']),
                        os.path.join(self.config['WORK_DIR'],
                                     self.config['SYSTEM_CONFIG']['Ingest']['Video']))

        self.log.info("Initializing IQR Blueprint -- Video")
        self.module_vsearch = IQRSearch('VideoSearch', self, ingest_video,
                                        'ColorDescriptor_Video_csift',
                                        'SVMIndexer_HIK',
                                        url_prefix="/vsearch")
        self.register_blueprint(self.module_vsearch)

        self.log.info("Initializing IQR Blueprint -- Image")
        self.module_isearch = IQRSearch('ImageSearch', self, ingest_image,
                                        'ColorDescriptor_Image_csift',
                                        'SVMIndexer_HIK',
                                        url_prefix="/isearch")
        self.register_blueprint(self.module_isearch)

        #
        # Basic routing
        #

        @self.route('/home')
        @self.route('/')
        def smqtk_index():
            self.log.info("Session: %s", flask.session.items())
            return flask.render_template("index.html")

    def run(self, host=None, port=None, debug=False, **options):
        """
        Override of the run method, drawing running host and port from
        configuration by default. 'host' and 'port' values specified as argument
        or keyword will override the app configuration.
        """
        super(SMQTKSearchApp, self).run(host=(host or self.config['RUN_HOST']),
                                        port=(port or self.config['RUN_PORT']),
                                        **options)
