# SMQTK - Deprecated
![CI Unittests](https://github.com/Kitware/SMQTK/workflows/CI%20Unittests/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/smqtk/badge/?version=latest)](https://smqtk.readthedocs.io/en/latest/?badge=latest)

## Deprecated
As of Jan 2021, SMQTK v0.14.0 has been deprecated. The various interfaces and implementations have been broken out into the following distinct packages which will continue to be supported instead.

* [SMQTK-Core](https://github.com/Kitware/SMQTK-Core) provides underlying tools used by other libraries.

* [SMQTK-Dataprovider](https://github.com/Kitware/SMQTK-Dataprovider) provides data structure abstractions.

* [SMQTK-Image-IO](https://github.com/Kitware/SMQTK-Image-IO) provides interfaces and implementations around image input/output.

* [SMQTK-Descriptors](https://github.com/Kitware/SMQTK-Descriptors) provides algorithms and data structures around computing descriptor vectors.

* [SMQTK-Classifier](https://github.com/Kitware/SMQTK-Classifier) provides interfaces and implementations around classification.

* [SMQTK-Detection](https://github.com/Kitware/SMQTK-Detection) provides interfaces and support for black-box object detection.

* [SMQTK-Indexing](https://github.com/Kitware/SMQTK-Indexing) provides interfaces and implementations around the k-nearest-neighbor algorithm.

* [SMQTK-Relevancy](https://github.com/Kitware/SMQTK-Relevancy) provides interfaces and implementations around providing search relevancy estimation.

* [SMQTK-IQR](https://github.com/Kitware/SMQTK-IQR) provides classes and utilities to perform the Interactive Query Refinement (IQR) process.

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
