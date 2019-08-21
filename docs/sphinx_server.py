#!/usr/bin/env python

"""
This module is designed to used with _livereload to 
make it a little easier to write Sphinx documentation.
Simply run the command::
    python sphinx_server.py

and browse to http://localhost:5500

livereload_: https://pypi.python.org/pypi/livereload
"""
import os

from livereload import Server, shell

rebuild_cmd = shell('make html', cwd='.')

watch_dirs = [
    '.',
    'architecture',
    'examples', 'examples/nnss_incremental_update',
    'quickstarts',
    'release_notes',
    'webservices',
]

watch_globs = [
    '*.rst', '*.ipynb'
]

server = Server()
server.watch('conf.py', rebuild_cmd)
# Cover above configured watch dirs and globs matrix.
for d in watch_dirs:
    for g in watch_globs:
        server.watch(os.path.join(d, g), rebuild_cmd)
# Watch source python files.
os.path.walk('../python/smqtk',
             lambda arg, dirname, fnames: server.watch(dirname+'/*.py',
                                                       rebuild_cmd),
             None)
# Optionally change to host="0.0.0.0" to make available outside localhost.
server.serve(root='_build/html')
