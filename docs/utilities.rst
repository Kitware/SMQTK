
Utilities and Applications
--------------------------

Also part of SMQTK are support utility modules, utility scripts (effectively the "binaries") and service-oriented and demonstration web applications.

Utility Modules
^^^^^^^^^^^^^^^

Various unclassified functionality intended to support the primary goals of SMQTK.
See doc-string comments on sub-module classes and functions in [``smqtk.utils``](/python/smqtk/utils) module.

Utility Scripts
^^^^^^^^^^^^^^^

Located in the [``bin``](/bin) directory are various scripts intended to provide quick access or generic entry points to common SMQTK functionality.
These scripts generally require configuration via a JSON text file.
By rule of thumb, scripts that require a configuration also provide an option for outputting a default or example configuration file.

Currently available utility scripts:

computeDescriptor
+++++++++++++++++

.. argparse::
   :ref: computeDescriptor.cli_parser
   :prog: computeDescriptor

createFileIngest
++++++++++++++++

.. argparse::
   :ref: createFileIngest.cli_parser
   :prog: createFileIngest

removeOldFiles
++++++++++++++

.. argparse::
   :ref: removeOldFiles.cli_parser
   :prog: removeOldFiles

* [``runApplication.py``](/bin/runApplication.py)
    * Generic entry point for running SMQTK web applications defined in [``smqtk.web``](/python/smqtk/web).
