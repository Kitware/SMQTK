SMQTK Pending Release Notes
===========================


Updates / New Features
----------------------


Fixes
-----
Docker
* Fix issue with IQR playground image where matplotlib was attempting to use
  the TkAgg backend by default by adding a ``matplotlibrc`` file to specify the
  use of the ``Agg`` backend.

Misc
* Update requirements versions for: Flask, Flask-Cors
* Update Travis-CI conviguration to assume less default values.
