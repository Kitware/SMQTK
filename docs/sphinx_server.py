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
server = Server()
server.watch('*.rst', shell('make html', cwd='.'))
server.watch('architecture/*.rst', shell('make html', cwd='.'))
server.watch('examples/*.rst', shell('make html', cwd='.'))
server.watch('examples/nnss_incremental_update/*.rst', shell('make html', cwd='.'))
server.watch('release_notes/*.rst', shell('make html', cwd='.'))
server.watch('webservices/*.rst', shell('make html', cwd='.'))
server.watch('conf.py', shell('make html', cwd='.'))
# Watch source python files.
os.path.walk('../python/smqtk',
             lambda arg, dirname, fnames: server.watch(dirname+'/*.py',
                                                       shell('make html',
                                                             cwd='.')),
             None)
# Optionally change to host="0.0.0.0" to make available outside localhost.
server.serve(root='_build/html')
