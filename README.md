# SMQTK

# Dependencies
In order to provide complete functionality, the following dependencies are required:

* ColorDescriptor
  * For CSIFT, TCH, etc. feature descriptors.
  * http://koen.me/research/colordescriptors/
  * After unpacking the downloaded ZIP archive, add the directory it was
    extracted to to the PYTHONPATH so the DescriptorIO.py module can be
    accessed and used within the SMQTK library.
  * Note that a license is required for commercial use (See the koen.me
    webpage).
* MongoDB
  * MongoDB is required for the Web application for session storage. This
    allows the Flask application API to modify session contents when within
    AJAX routines. which is sometimes required for asynchronous user state
    interaction/modification.
  * This is not a perminent requirement as other mediums can be used for this
    purpose, however they would need implementation.

## Recommended
Before installing Numpy and Scipy, it may be useful or required to install
BLAS or LAPACK libraries for certain functionalities and efficiency.

# Forming a new ingest

## Create File "Ingest"
The first step to creating an formal ingest is to aggregate the base files into
a canonical location and in a known (to the system) layout.

High level steps:
* Configure location and component pathing in the ``etc/system_config.json``
  file. See the example configurations in the file to get an idea of what to
  write.
* Call ``bin/CreateIngest.py`` script to add files to an ingest defined in the
  ``etc/system_config.json`` file.

### The ``system_config.json`` file
In order for the system to know where to put and look for files, we need to
supply it with such information. This configuration file declares incrementally
relative paths for system components that are ingest (or data) dependent.

Previous iterations of this software used static pathing and settings for
internal components, but this lead to a very rigid system that did not allow
for multiple ingests and component model/work files and settings to exist
side-by-side. This lead to more filesystem and source-code management than
desired, thus this configuration format was created.

### Creating the file ingest
TODO

### Generating model files for an ingest
TODO
