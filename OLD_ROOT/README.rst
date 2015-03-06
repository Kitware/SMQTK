 Introduction
============

This documents provides an installation guide for SMQTK online prototype system.
It enlists all the software prerequisite software libraries including databases.
It also provides pointers to the existing data, which will be used within the system.

Requirements
============

This manual is written with ubuntu 12.04 lts in mind, targeting a machine having at least 2 cores and 10 GB of ram.

Installation of dependencies
============================

To Install mongodb
------------------

Please follow instructions from - http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/

and install mongodb-10gen

A copy of working mongodb snapshot is included in final delivery folder as  data/smqtk-system/mongodb.tar.gz

The main system run script, described later in this README, requires the path to the database's data directory.
First, however, any running database using the default MongoDB port (27017) must be stopped.
If there is a mongod service on the system, which would occupy the default port, yhis can be done with the following command:

.. code-block:: shell-session

    $ sudo /etc/init.d/mongodb stop

TODO: Description of how to ingest a new dataset into the current database format.

Python and required python packages
-----------------------------------

Requires python 2.7x, and pip.

Click here to install `python <http://www.python.org/download/>`_ and `pip <http://stackoverflow.com/questions/4750806/how-to-install-pip-on-windows>`_ on windows
Also please make sure that the python and scritps ("c:/python27/Scripts") are in PATH environment variable.

To install pip and other dependencies

.. code-block:: shell-session

   $ sudo apt-get install python-pip
   $ sudo apt-get install python-dev
   $ sudo apt-get install python-docutils
   $ sudo apt-get install python-numpy
   $ sudo apt-get install python-scipy
   $ sudo apt-get install python-matplotlib


Following dependencies are listed in **requirements.txt**

- flask
- pymongo
- sphinx
- flask-login
- pillow
- gevent

To install them -

.. code-block:: shell-session

   $ pip install -r requirements.txt --upgrade

There some packages like PIL and Scipy which do require dependencies to compile, and pip install will fail if these are
not installed. Those should be installed by other mechanisms, i.e. by using apt-get or distribution packages for
specific platform.



Scipy Hierarchial clustering needs some patching (TODO: can be avoided by rewriting that function in the code base)

.. note::

    if additional experimental modules are enabled, they will impose more dependencies like

        - suggestions : nltk

Included third-party libraries
------------------------------

Included in the Backend module is a TPL directory containing bundled third
-party packages:

  - libSVM 3.17
  - flann 1.8.4

The compilation and installation of these are managed by the CMake build system.

Before running any part of the system, source either the build or installed
setup script to ensude that bundled materials are used correctly:

.. code-block:: shell-session

    $ ${BUILD_DIR}/setup_env.build.sh

if using the build directory, or the following in you're using an installed version:

.. code-block:: shell-session

    $ ${INSTALL_DIR}/setup_smqtk.sh

Static video data
-----------------

Entire collection of video clips encoded using ogg-vorbis format for serving over web are provided in data/smqtk-system/clips.tar

The encoded video clips should be extracted to smqtk-system/static/data/clips

Configuration setting in the smqtk_config.py (or the config file that you are applying configuration from) should be:

.. code-block:: python

    VIDEO_URL_PREFIX ="http://127.0.0.1:5001/static/data/clips/"

This path is used by a local static-files server to provide the main application with said static data.

Data Matrices
=============

Data matrices are provided at: data/smqtk-system/backend_data.tgz

The contents should be extracted to smqtk-system/data/


To build this documenatation
============================

Detailed documentation can be built using sphinx

.. code-block:: shell-session

   $ cd docs
   $ make html

The html documentation can be seen starting at Docs/_build/html/index.html

To build individual rst files
-----------------------------

rst2pdf utility can create pdf documentation with a single command

.. code-block:: shell-session

   $ rst2html index.html

rst2html utility can create html documentation

.. code: shell-session

   $ rst2html index.html

However the generated file does not include css required render syntax highlighting. The css file can be
created by using pygments package

.. code-block:: shell-session

   pygmentize -S default -f html -a .highlight > style.css

After the css is created it should be loaded by the generated html. This can be achieved by manually inserting following
 line in the head tag.

.. code-block:: html

   <link rel="stylesheet" href="style.css"> </link>

To run the webserver
====================

A one-button python script is provided to manage and run the various moving parts
of the current system. After sourcing the appropriate setup file (depending on
whether you're using a build or install environment), the run_smqtk.py script
may be run:

.. code-block:: shell-session:

    $ ./run_smqtk.py --dbpath DIRPATH

The ``--dbpath`` option used above is required and should point to the MongoDB
data directory containing your dataset's ingested contents.


Username and Passwords
----------------------

The login system is currently simple and stored in a python dictionary (for fast implementation with no time for adapting other systems).
That file needs to be applied separately.

The user logins can be changed by editing the local_config.py file in the repository


Deploy
======

Example for deploying flask applications with fabric is given at - http://flask.pocoo.org/docs/patterns/fabric/

An example configuration for deployment using apache mod_wsgi is as follows

Sample Apache configuration
---------------------------

.. code-block:: none

  <VirtualHost admin.slide-atlas.org:80>
       ServerName smqtk.localhost
       ServerAdmin admin@localhost

       WSGIDaemonProcess smqtk user=www-data group=www-data threads=1
       WSGIScriptAlias / /var/www/run_apache.wsgi

       <Directory /var/www>
           WSGIProcessGroup smqtk
           WSGIApplicationGroup %{GLOBAL}
           Order deny,allow
           Allow from all
       </Directory>

        ErrorLog ${APACHE_LOG_DIR}/error.log

        # Possible values include: debug, info, notice, warn, error, crit,
        # alert, emerg.
        LogLevel warn

        CustomLog ${APACHE_LOG_DIR}/access.log combined

  </VirtualHost>
