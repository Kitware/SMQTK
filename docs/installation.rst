Installation
============

There are two ways to get ahold of SMQTK.
The simplest is to install via the :command:`pip` command.
Alternatively, the source tree can be acquired and build/install SMQTK via CMake or ``setuptools``.


From :command:`pip`
-------------------
In order to get the latest version of SMQTK from PYPI:

.. prompt:: bash

    pip install --upgrade smqtk

This method will install all of the same functionality as when installing from source, but not as many plugins will be functional right out of the box.
This is due to some plugin dependencies not being installable through pip.
We will see more on this in the section below.

Extras
^^^^^^
A few extras are defined for the ``smqtk`` package:

- ``docs``
    - Dependencies for building SMQTK documentation.
- ``caffe``
    - Minimum required packages for when using with the Caffe plugin.
- ``flann``
    - Required packages for using FLANN-based plugins.
    - There is not an adequate version in the standard PYPI repository (>=1.8.4).
      For FLANN plugin functionality, it is recommended to either use your system package manager or SMQTK from source.
- ``postgres``
    - Required packages for using PostgreSQL-based plugins.
- ``solr``
    - Required packages for using Solr-based plugins.


From Source
-----------
Acquiring and building from source is different than installing from :command:`pip` because:

- Includes FLANN and libSVM [#customLibSvm]_ libraries and (patched) python bindings in the CMake build.
  CMake installation additionally installs these components
- CPack packaging support (make RPMs, etc.). [#CpackIncomplete]_

The inclusion of FLANN and libSVM in the source is generally helpful due to their lack of [up-to-date] availability in the PYPI and system package repositories.
When available via a system package manager, it is often not easy to use when dealing with a virtual environment (e.g. virtualenv or Anaconda).

The sections below will cover the quick-start steps in more detail:

* :ref:`installation-fromSource-SystemDependencies`
* :ref:`installation-fromSource-GettingTheSource`
* :ref:`installation-fromSource-InstallingPythonDeps`
* :ref:`installation-fromSource-cmakeBuild`
* :ref:`installation-fromSource-buildDocumentation`


Quick Start
^^^^^^^^^^^

.. prompt:: bash

    # Check things out
    cd /where/things/should/go/
    git clone https://github.com/Kitware/SMQTK.git source
    # Install python dependencies to environment
    pip install -r source/requirements.txt
    # SMQTK build
    mkdir build
    pushd build
    cmake ../source
    make -j2
    popd
    # Set up SMQTK environment by sourcing file
    . build/setup_env.build.sh
    # Running tests
    python source/setup.py test


.. _installation-fromSource-SystemDependencies:

System dependencies
^^^^^^^^^^^^^^^^^^^
In order retrieve and build from source, your system will need at a minimum:

* git
* cmake >=2.8
* c++ compiler (e.g. gcc, clang, MSVC etc.)

In order to run the provided IQR-search web-application, introduced later when describing the provided web services and applications, the following system dependencies are additionally required:

* MongoDB [#MongoDep]_


.. _installation-fromSource-GettingTheSource:

Getting the Source
^^^^^^^^^^^^^^^^^^
The SMQTK source code is currently hosted `on GitHub here <https://github.com/Kitware/SMQTK>`_.

To clone the repository locally:

.. prompt:: bash

    git clone https://github.com/Kitware/SMQTK.git /path/to/local/source


.. _installation-fromSource-InstallingPythonDeps:

Installing Python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After deciding and activating what environment to install python packages into (system or a virtual), the python dependencies should be installed based on the :file:`requirements.*.txt` files found in the root of the source tree.
These files detail different dependencies, and their exact versions tested, for different components of SMQTK.

The the core required python packages are detailed in: :file:`requirements.txt`.

In addition, if you wish to be able to build the Sphinx_ based documentation for the project: :file:`requirements.docs.txt`.
These are separated because not everyone wishes or needs to build the documentation.

Other optional dependencies and what plugins they correspond to are found in: :file:`requirements.optional.txt`

Note that if :command:`conda` [#conda]_ is being used, not all packages listed in our requirements files may be found in :command:`conda`'s repository.

Installation of python dependencies via pip will look like the following:

.. prompt:: bash

    pip install -r requirements.txt [-r requirements.docs.txt]

Where the :file:`requirements.docs.txt` argument is only needed if you intend to build the SMQTK documentation.


Building NumPy and SciPy
""""""""""""""""""""""""
If NumPy and SciPy is being built from source when installing from :command:`pip`, either due to a wheel not existing for your platform or something else, it may be useful or required to install BLAS or LAPACK libraries for certain functionality and efficiency.

Additionally, when installing these packages using :command:`pip`, if the :envvar:`LDFLAGS` or :envvar:`CFLAGS`/:envvar:`CXXFLAGS`/:envvar:`CPPFLAGS` are set, their build may fail as they are assuming specific setups [#NumpyScipyBuild]_.


Additional Plugin Dependencies
""""""""""""""""""""""""""""""
Some plugins in SMQTK may require additional dependencies in order to run, usually python but sometimes not.
In general, each plugin should document and describe their specific dependencies.

For example, the ColorDescriptor implementation required a 3rd party tool to downloaded and setup.
Its requirements and restrictions are documented in :file:`python/smqtk/algorithms/descriptor_generator/colordescriptor/INSTALL.md`.


.. _installation-fromSource-cmakeBuild:

CMake Build
^^^^^^^^^^^
See the example below for a simple example of how to build SMQTK

Navigate to where the build products should be located.
It is recommended that this not be the source tree.
Build products include some C/C++ libraries, python modules and generated scripts.

If the desired build directory, and run the following, filling in ``<...>`` slots with appropriate values:

.. prompt:: bash

    cmake <source_dir_path>

Optionally, the `ccmake` command line utility, or the GUI version, may be run in order to modify options for building additional modules.
Currently, the selection is very minimal, but may be expanded over time.


.. _installation-fromSource-buildDocumentation:

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the documentation for SMQTK is maintained as a collection of `reStructuredText_` documents in the :file:`docs` folder of the project.
This documentation can be processed by the :program:`Sphinx` documentation tool into a variety of documentation formats, the most common of which is HTML.

Within the :file:`docs` directory is a Unix :file:`Makefile` (for Windows systems, a :file:`make.bat` file with similar capabilities exists).  This :file:`Makefile`
takes care of the work required to run :program:`Sphinx` to convert the raw documentation to an attractive output format.  For example::

    make html

Will generate HTML format documentation rooted a :file:`docs/_build/html/index.html`.

The command::

    make help

Will show the other documentation formats that may be available (although be aware that some of them require additional dependencies such as :program:`TeX` or :program:`LaTeX`.)


Live Preview
""""""""""""

While writing documentation in a mark up format such as ``reStructuredText`` it is very helpful to be able to preview the formatted version of the text.
While it is possible to simply run the ``make html`` command periodically, a more seamless version of this is available.
Within the :file:`docs` directory is a small Python script called :file:`sphinx_server.py`.
If you execute that file with the following command::

    python sphinx_server.py

It will run small process that watches the :file:`docs` folder for changes in the raw documentation :file:`*.rst` files and re-runs :command:`make html` when changes are detected.
It will serve the resulting HTML files at http://localhost:5500.
Thus having that URL open in a browser will provide you with a relatively up to date preview of the rendered documentation.


.. rubric:: Footnotes
.. [#customLibSvm] Included libSVM is a customized version based on v3.1
.. [#CpackIncomplete] These features are largely still in development and may not work correctly yet.
.. [#MongoDep] This requirement will hopefully go away in the future, but requires an alternate session storage implementation.
.. [#conda] For more information on the :command:`conda` command and system, see the `Conda documentation`_.
.. [#NumpyScipyBuild] This may have changed since wheels were introduced.


.. _Sphinx: http://sphinx-doc.org/
.. _Conda documentation:  http://conda.pydata.org/docs/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
