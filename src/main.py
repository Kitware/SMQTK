#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cherrypy
import girder.utility.config
import girder.utility.server
import mako.template
import os

import geospace

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PACKAGE_DIR)

class GeoAppRoot(object):
    """
    Serve the root webpage for our application.
    """
    exposed = True
    indexHtml = None
    vars = {
        'apiRoot': 'api/v1',
        'staticRoot': 'built',
        'girderRoot': 'girder/static',
    }

    def GET(self):
        if self.indexHtml is None:
            page = open(os.path.join(ROOT_DIR, 'index.html')).read()
            print '%r' % page
            self.indexHtml = mako.template.Template(page).render(**self.vars)
        return self.indexHtml


class GeoApp():
    def __del__(self):
        cherrypy.engine.exit()

    """Start the server and serve until stopped."""
    def start(self):
        #cherrypy.config['database']['uri'] = 'mongodb://localhost:27017/geoapp'
        cherrypy.config['server.socket_port'] = 8001
        self.root = GeoAppRoot()
        # Create the girder services and place them at /girder
        self.root.girder, appconf = girder.utility.server.configureServer()
        curConfig = girder.utility.config.getConfig()
        localappconf = {
            '/src': {
                'tools.staticdir.on': 'True',
                'tools.staticdir.dir': os.path.join(ROOT_DIR, 'src')
            },
            'girder/static': curConfig['/static'],
            'girder/static/lib/bootstrap/fonts': {
                'tools.staticdir.on': 'True',
                'tools.staticdir.dir': os.path.join(
                    ROOT_DIR, 'built/lib/bootstrap/fonts')
            }
        }
        appconf.update(localappconf)
        curConfig.update(localappconf)

        self.server = cherrypy.tree.mount(self.root, '/', appconf)
        # move the girder API from /girder/api to /api
        self.root.api = self.root.girder.api
        del self.root.girder.api

        self.root.girder.updateHtmlVars({'staticRoot': 'girder/static'})
        self.root.api.v1.updateHtmlVars({'staticRoot': 'girder/static'})

        info = {
            'config': appconf,
            'serverRoot': self.root,
            'apiRoot': self.root.api.v1
        }
        # load plugin is called with plugin, root, appconf, root.api.v1 as
        #   apiRoot, curConfig
        # the plugin module is then called with info = {name: plugin,
        #   config: appconf, serverRoot: root, apiRoot: root.api.v1,
        #   pluginRootDir: (root)}
        # it can modify root, appconf, and apiRoot

        geospace.load(info)

        cherrypy.engine.start()
        cherrypy.engine.block()

if __name__ == '__main__':
    app = GeoApp()
    app.start()
