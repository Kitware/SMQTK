#!/usr/bin/env sh
nosetests --with-doctest --with-coverage --cover-package=smqtk --exclude-dir-file=nose_exclude_dirs.txt python/smqtk
