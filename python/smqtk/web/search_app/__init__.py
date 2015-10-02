"""
Top level flask application
"""

import flask
import os.path

from smqtk.utils import DatabaseInfo
from smqtk.utils.configuration import merge_configs
from smqtk.utils.mongo_sessions import MongoSessionInterface
from smqtk.web import SmqtkWebApp

from .modules.login import LoginMod
from .modules.iqr import IqrSearch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class IqrSearchApp (SmqtkWebApp):

    @classmethod
    def get_default_config(cls):
        c = super(IqrSearchApp, cls).get_default_config()
        merge_configs(c, {
            "mongo": {
                "server": "127.0.0.1:27017",
                "database": "smqtk",
            },
            # Each entry in this mapping generates a new tab in the GUI
            "iqr_tabs": [
                IqrSearch.get_default_config(),
            ]
        })
        return c

    @classmethod
    def from_config(cls, config_dict):
        return cls(config_dict)

    def __init__(self, json_config):
        super(IqrSearchApp, self).__init__(json_config)

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

        # Navigable blueprints. This should contain the blueprints that a user
        # should be able to navigate to. Not all blueprints have navigable
        # content or should allow user explicit navigation to, thus this
        # structure.
        #: :type: list of flask.Blueprint
        self._navigable_blueprints = []

        # Login module
        self.log.info("Initializing Login Blueprint")

        self.module_login = LoginMod('login', self)
        self.register_blueprint(self.module_login)

        # IQR modules
        # - for each entry in 'iqr_tabs', initialize a separate IqrSearch
        #   instance.
        self._iqr_search_modules = []
        for iqr_search_config in self.json_config['iqr_tabs']:
            self.log.info("Initializing IQR tab '%s'",
                          iqr_search_config['name'])
            self.log.debug("IQR tab config:\n%s", iqr_search_config)
            m = IqrSearch.from_config(iqr_search_config, self)
            self.register_blueprint(m)
            self.add_navigable_blueprint(m)
            self._iqr_search_modules.append(m)

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


APPLICATION_CLASS = IqrSearchApp
