# SMQTK
![Build Status](https://travis-ci.org/Kitware/SMQTK.svg?branch=master)

[![Documentation Status](https://readthedocs.org/projects/smqtk/badge/?version=latest)](https://smqtk.readthedocs.io/en/latest/?badge=latest)

## Intent
Social Multimedia Query ToolKit aims to provide a simple and easy to use API for:

* Scalable data structure interfaces and implementations, with a focus on those relevant for machine learning.
* Algorithm interfaces and implementations of machine learning algorithms with a focus on media-based functionality.
* High-level applications and utilities for working with available algorithms and data structures for specific purposes.

Through these features, users and developers are able to access various machine learning algorithms and techniques to use over different types of data for different high level applications.
Examples of high level applications may include being able to search a media corpus for similar content based on a query, or providing a content-based relevancy feedback interface for a web application.

## Documentation

Documentation for SMQTK is maintained at
[ReadtheDocs](https://smqtk.readthedocs.org), including
[build instructions](https://smqtk.readthedocs.io/en/latest/installation.html)
and [examples](https://smqtk.readthedocs.org/en/latest/examples/overview.html).

Alternatively, you can build the sphinx documentation locally for the most
up-to-date reference (see also: [Building the Documentation](
https://smqtk.readthedocs.io/en/latest/installation.html#building-the-documentation)):
```bash
# Navigate to the documentation root.
cd docs
# Install dependencies and build Sphinx docs.
pip install -r readthedocs-reqs.txt
make html
# Open in your favorite browser!
firefox _build/html/index.html
```
