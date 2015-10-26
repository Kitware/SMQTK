Building SMQTK
==============

Dependencies
------------

In order to provide basic functionality:

* Build SMQTK via CMAKE.
 * Currently, a naive CMake configuration (no modifications to options) is acceptable for basic functionality.
* Install python packages detailed in the :file:`requirements.*.txt` files.

In order to run provided SMQTKSearchApp web application, the following are additionally required:

* MongoDB
 * MongoDB is required for the Web application for session storage.
   This allows the Flask application API to modify session contents when within AJAX routines.
   This required for asynchronous user-session state interaction/modification.
 * This is not a permanent requirement as other mediums can be used for this purpose, however they would need implementation.

Installing Python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two files that list required python packages:

* :file:`requirements.conda.txt`
* :file:`requirements.pip.txt`

Required packages have been split up this way because conda does not provide all packages that pip can.
While conda is generally considered the preferred method of acquiring python dependencies due to their pre-built nature, some of our requirements are not available through conda.

Installing with Conda and Pip
"""""""""""""""""""""""""""""
The three-step python dependency installation using both conda and pip will look like the following:

.. prompt:: bash

    conda create -n <env_name> --file requirements.conda.txt`
    . activate <env_name>`
    pip install -r requirements.pip.txt`


Installing with just Pip
""""""""""""""""""""""""

If installation of python dependencies via pip only is desired, or if local compilation of packages is desired, the following is recommended:

.. prompt:: bash

    pip install -r requirements.conda.txt -r requirements.pip.txt

NumPy and SciPy
+++++++++++++++

If installing NumPy and SciPy via pip, it may be useful or required to install BLAS or LAPACK libraries for certain functionality and efficiency.

Additionally, when installing these packages using :command:`pip`, if the :envvar:`LDFLAGS` or :envvar:`CFLAGS`/:envvar:`CXXFLAGS`/:envvar:`CPPFLAGS` are set, their build may fail as they are assuming specific setups.

Additional Descriptor Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Descriptors implemented in SMQTK may require additional dependencies in order to run.
This may be because a descriptor required additional libraries or tools on the system.
For example, the ColorDescriptor implementation required a 3rd party tool to downloaded and setup.

* ColorDescriptor
 * For CSIFT, TCH, etc. feature descriptors.
 * http://koen.me/research/colordescriptors/
 * After unpacking the downloaded ZIP archive, add the directory it was extracted to to the PYTHONPATH so the DescriptorIO.py module can be accessed and used within the SMQTK library.
 * Note that a license is required for commercial use (See the koen.me webpage).

As more descriptors are added, more optional dependencies may be introduced.


Build
-----

Building SMQTK requires CMake and a C/C++ compiler.
See the example below for a simple example of how to build SMQTK

CMake Build
^^^^^^^^^^^

Navigate to where the build products should be located.
It is recommended that this not be the source tree.
Build products include some C/C++ libraries, python modules and generated scripts.

If the desired build directory, and run the following, filling in ``<...>`` with appropriate values:

.. prompt:: bash

    $ cmake <source_dir_path>`

Optionally, the `ccmake` command line utility, or the GUI version, may be run in order to modify options for building additional modules.
Currently, the selection is very minimal, but may be expanded over time.
 
Example
-------

.. prompt:: bash

    # Check things out
    cd /where/things/should/go/
    git clone https://github.com/Kitware/SMQTK.git source
    # Install python dependencies to environment
    pip install -r source/requirements.conda.txt -r source/requirements.pip.txt
    # SMQTK build
    mkdir build
    pushd build
    cmake ../source
    make -j2
    popd
    # Set up SMQTK environment by sourcing file
    . build/setup_env.build.sh
    # Running tests
    source/run_tests.sh
    
