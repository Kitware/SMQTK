SMQTK Architecture Overview
===========================

SMQTK provides at its lowest level semantics for plugins and configuration.
These are provided by some utility functions and two mixin classes:
:py:class:`smqtk.utils.plugin.Pluggable` and
:py:class:`smqtk.utils.configuration.Configurable`.
These are explained further in the "Plugins and Configuration" section.

Subsequent to these two mixin classes, SMQTK provides two main categories of
interfaces: algorithms and data representations.
This organization of philosophy roughly aligns with the concept of data
oriented design.
Algorithms are usually interfaces defining a behavioral or transformative
action(s), abstracting away how that behavior or transformation is achieved.
Data representation interfaces define the encapsulation of some data structure,
abstracting away where that data is stored..

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
