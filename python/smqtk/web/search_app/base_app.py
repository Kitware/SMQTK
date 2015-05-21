"""
Top level flask application
"""

import flask
import logging
import os.path

from smqtk.utils import DatabaseInfo, SimpleTimer
from smqtk.utils.mongo_sessions import MongoSessionInterface


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SMQTKSearchApp (flask.Flask):

    # Optional environment variable that can point to a configuration file
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

        # Navigable blueprints. This should contain the blueprints that a user
        # should be able to navigate to. Not all blueprints have navigable
        # content or should allow user explicit navigation to, thus this
        # structure.
        #: :type: list of flask.Blueprint
        self._navigable_blueprints = []

        # Login module
        self.log.info("Initializing Login Blueprint")
        from .modules.login import LoginMod
        self.module_login = LoginMod('login', self)
        self.register_blueprint(self.module_login)

        # IQR modules
        # TODO: At the moment, for simplicity, we're fixing the feature detector
        #       and indexer types. In the future this should either be moved
        #       to something that can be chosen by the user or a
        #       multi-feature/indexer fusion system.
        from .modules.iqr import IQRSearch, IQRSearchFusion

        with SimpleTimer("Loading Example Image ingest + IQR...", self.log.info):
            ic_example_image = IngestConfiguration("example_image")
            self.mod_example_image = IQRSearch(
                "Image Search - Example Imagery",
                self, ic_example_image,
                "ColorDescriptor_Image_csift", "SVMIndexer_HIK",
                url_prefix='/image_example'
            )
            self.register_blueprint(self.mod_example_image)
            self.add_navigable_blueprint(self.mod_example_image)

        with SimpleTimer("Loading Example Image ingest + IQR Fusion", self.log.info):
            self.mod_example_image_fusion = IQRSearchFusion(
                "Image Search Fusion - Example Imagery",
                self, ic_example_image,
                "Average",
                url_prefix='/image_example_fusion'
            )
            self.register_blueprint(self.mod_example_image_fusion)
            self.add_navigable_blueprint(self.mod_example_image_fusion)

        with SimpleTimer("Loading Example Video ingest + IQR...", self.log.info):
            ic_example_video = IngestConfiguration("example_video")
            self.mod_example_video = IQRSearch(
                "Video Search - Example Videos",
                self, ic_example_video,
                "ColorDescriptor_Video_csift", "SVMIndexer_HIK",
                url_prefix='/video_example'
            )
            self.register_blueprint(self.mod_example_video)
            self.add_navigable_blueprint(self.mod_example_video)

        #
        # Basic routing
        #

        @self.route('/home')
        @self.route('/')
        def smqtk_index():
            self.log.info("Session: %s", flask.session.items())
            # noinspection PyUnresolvedReferences
            return flask.render_template("index.html", **self.nav_bar_content())

    def add_navigable_blueprint(self, bp):
        """
        Register a navigable blueprint. This is not the same thing as
        registering a blueprint with flask, which should happen separately.

        :param bp: Blueprint to register as navigable via the navigation bar.
        :type bp: flask.Blueprint

        """
        self._navigable_blueprints.append(bp)

    def nav_bar_content(self):
        """
        Formatted dictionary for return during a flask.render_template() call.
        This content must be included in all flask.render_template calls that
        are rendering a template that descends from our ``base.html`` template
        in order to allow proper construction and rendering of navigation bar
        content.

        For example, when returning a flask.render_template() call:
        >> ret = {"things": "and stuff"}
        >> ret.update(smqtk_search_app.nav_bar_content())
        >> return flask.render_template("some_template.tmpl", **ret)

        :return: Dictionary of content required for proper display of the
            navigation bar. Contains keys of module names and values of module
            URL prefixes.
        :rtype: {"nav_content": list of (tuple of str)}
        """
        l = []
        for nbp in self._navigable_blueprints:
            l.append((nbp.name, nbp.url_prefix))
        return {
            "nav_content": l
        }

    def run(self, host=None, port=None, debug=False, **options):
        """
        Override of the run method, drawing running host and port from
        configuration by default. 'host' and 'port' values specified as argument
        or keyword will override the app configuration.
        """
        super(SMQTKSearchApp, self).run(host=(host or self.config['RUN_HOST']),
                                        port=(port or self.config['RUN_PORT']),
                                        **options)
