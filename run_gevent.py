#!/usr/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
from gevent import monkey
monkey.patch_all()

from flask import Flask
import gevent.wsgi
import gevent.monkey
import werkzeug.serving

from WebUI import app
app.debug = True

# @werkzeug.serving.run_with_reloader
def run_server():
    ws = gevent.wsgi.WSGIServer(listener=('0.0.0.0', 5003),
                                application=app)
    ws.serve_forever()

if __name__ == "__main__":
    run_server()
    from WebUI.query_refinement import backend
    backend.shutdown()


