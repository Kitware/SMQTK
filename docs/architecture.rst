SMQTK Architecture Overview
===========================

SMQTK provides a plugin infrastructure to allow the definition of high level
interfaces and allow a number of implementations to fulfil those interfaces.

Within this architecture, SMQTK provides two main categories of interfaces:
algorithms and data representations.
Algorithms are usually interfaces defining a functional process where as data
representation interfaces define the encapsulation of some data structure.

Building upon algorithm and data representation interfaces, there is a
sub-module providing some general web services: :mod:`smqtk.web`.
Of likely interest is headless IQR web-service
(:py:mod:`smqtk.web.iqr_service`).
There is also a demonstration IQR web application with a simple UI as well as
other headless web services (:py:mod:`smqtk.web.search_app`).


.. toctree::
  :maxdepth: 2

  architecture/plugins_configuration
  dataabstraction
  algorithms
  webservices
  utilities
