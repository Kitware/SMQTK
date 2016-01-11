.. _webapplabel:
Web Service and Demonstration Applications
==========================================

Included in SMQTK are a few web-based service and demonstration applications, providing a view into the functionality provided by SMQTK algorithms and utilities.

.. _run_application:
runApplication
--------------

This script can be used to run any conforming (derived from `SmqtkWebApp`) SMQTK web based application.
Web services should be runnable via the ``bin/runApplication.py`` script.

.. argparse::
   :ref: runApplication.cli_parser
   :prog: runApplication

SmqtkWebApp
-----------
This is the base class for all web applications and services in SMQTK.

.. autoclass:: smqtk.web.SmqtkWebApp
   :members:


Sample Web Applications
-----------------------

.. toctree::
   :maxdepth: 3

   webservices/descriptorservice
   webservices/iqrdemonstration

