#! /usr/bin/env sh
nosetests --with-doctest --with-coverage --cover-package=smqtk --nocapture python/smqtk --exclude-dir-file=nose_exclude_dirs.txt
