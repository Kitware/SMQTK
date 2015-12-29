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

In addition, if you wish to be able to build the Sphinx_ based documentation for the project there are two additional dependency files:

* :file:`requirements.docs.conda.txt`
* :file:`requirements.docs.pip.txt`

Required packages have been split up this way because conda does not provide all packages that pip can.  Further, not everyone wishes or needs to build the documentation.
While conda is generally considered the preferred method of acquiring python dependencies due to their pre-built nature, some of our requirements are not available through conda.

.. _Sphinx: http://sphinx-doc.org/

Installing with Conda and Pip
"""""""""""""""""""""""""""""

The three-step python dependency installation using both conda and pip will look like the following:

.. prompt:: bash

    conda create -n <env_name> --file requirements.conda.txt` [--file requirements.docs.conda.txt]
    . activate <env_name>
    pip install -r requirements.pip.txt [-r requirements.docs.pip.txt]


Where the :file:requirements.docs.*.txt arguments are only needed if you intend to build the SMQTK documentation.

Installing with just Pip
""""""""""""""""""""""""

If installation of python dependencies via pip only is desired, or if local compilation of packages is desired, the following is recommended:

.. prompt:: bash

    pip install -r requirements.conda.txt -r requirements.pip.txt [-r requirements.docs.conda.txt -r requirements.docs.pip.txt]

Where the :file:requirements.docs.*.txt arguments are only needed if you intend to build the SMQTK documentation.

NumPy and SciPy
^^^^^^^^^^^^^^^

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
"""""""

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
    
Building the Documentation
--------------------------

All of the documentation for SMQTK is maintained as a collection of `reStructuredText_` documents in the :file:`docs` folder of the project.  
This documentation can be processed by the :program:`Sphinx` documentation tool into a variety of documentation formats, the most common of which is HTML.

Within the :file:`docs` directory is a Unix :file:`Makefile` (for Windows systems, a :file:`make.bat` file with similar capabilities exists).  This :file:`Makefile` 
takes care of the work required to run :program:`Sphinx` to convert the raw documentation to an attractive output format.  For example::

    make html

Will generate HTML format documentation rooted a :file:`docs/_build/html/index.html`.

The command::

    make help

Will show the other documentation formats that may be available (although be aware that some of them require additional dependencies such as :program:`TeX` or :program:`LaTeX`.)

.. _reStructuredText: http://docutils.sourceforge.net/rst.html

Live Preview
^^^^^^^^^^^^

While writing documentation in a mark up format such as ``reStructuredText`` it is very helpful to be able to preview the formated version of the text.  While it is possible to 
simply run the ``make html`` command periodically, a more seamless version of this is available.  Within the :file:`docs` directory is a small Python script called 
:file:`sphinx_server.py`.   If you execute that file with the following command::

    python sphinx_server.py

It will run small process that watches the :file:`docs` folder for changes in the raw documentation :file:`*.rst` files and re-runs :command:`make html` when changes are detected.  It will
serve the resulting HTML files at http://localhost:5500.  Thus having that URL open in a browser will provide you with a relatively up to date preview of the rendered documentation.
