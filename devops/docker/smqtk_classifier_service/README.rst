Base Classifier Service Docker Image
====================================
This docker image provides a base for building turn-key classifier service
images.

This base image provides a SMQTK installation as well as an entrypoint to run
the classifier service with a fixed configuration file path.

Port 5002 is exposed and the configured server should use this port in any child
images. Make sure to configure the host to ``0.0.0.0`` for the service to be
accessible from outside the container.


Server Configuration
--------------------
Child images extending this should add their specific configuration and
model files in the ``/configuration`` directory inside the image. This directory
must contain the file ``server.json`` that will be used by the entrypoint
script as configuration file for the classification server.

An default example of this configuration file can be output by running::

    runApplication -l SmqtkClassifierService -g /path/to/output/file.json

A default configuration file is pre-installed that loads no classifier models
but can still be used with IQR session state files at runtime.
