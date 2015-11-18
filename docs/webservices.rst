Web Service and Demonstration Applications
==========================================

Included in SMQTK are a few web-based service and demonstration applications, providing a view into the functionality provided by SMQTK algorithms and utilities.

runApplication
--------------

This script can be used to run any conforming (derived from `SmqtkWebApp`) SMQTK web based application.
Web services should be runnable via the ``bin/runApplication.py`` script.

.. argparse::
   :ref: runApplication.cli_parser
   :prog: runApplication

SmqtkWebApp
-----------

.. autoclass:: smqtk.web.SmqtkWebApp
   :members:

Sample Web Applications
-----------------------

Descriptor Similarity Service
+++++++++++++++++++++++++++++

* Provides a web-accessible API for computing content descriptor vectors for available descriptor generator labels.
* Descriptor generators that are available to the service are based on the a configuration file provided to the server.

.. autoclass:: smqtk.web.descriptor_service.DescriptorServiceServer
   :members:

IQR Demo Application
++++++++++++++++++++

* Demo application for performing Interactive Query Refinement (IQR)
* Fully configurable 
* Requires algorithm models to be built

.. autoclass:: smqtk.web.search_app.IqrSearchApp
   :members:

