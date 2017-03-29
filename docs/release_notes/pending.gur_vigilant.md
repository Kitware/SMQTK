SMQTK Vigilant Release Notes
============================


Updates / New Features
----------------------

Algorithms

  Descriptor Generators

    * Added KWCNN DescriptorGenerator plugin

Devops

  * Added IQR / playground docker container setup. Includes:
    - CPU + NVIDIA GPU capable docker file.
    - Optional input image tiling.
    - Optional startup of RESTfule NN and IQR services.

  * Added initial Ansible roles for SMQTK and Caffe dependency.

Girder

  * Added initial simple Girder plugin to link to an external IQR webapp
    instance.

Representation

  Data Elements

    * Added plugin for Girder-hosted data elements

    * Added ``from_uri`` member function as well as global function to handle
      instance construction or selection via URI string specification.

    * Postgres data element will now automatically create its configured table
      if it doesn't exist and authentication and sufficient privileges.

  Descriptor Element

    * Postgres descriptor element will now automatically create its configured
      table if it doesn't exist and authentication and sufficient privileges.

  Descriptor Index

    * Postgres descriptor index will now automatically create its configured
      table if it doesn't exist and authentication and sufficient privileges.

Scripts

  * Added script to add GirderDataElements to a data set

Utilities

  * Started a module containing URL-base utility functions, initially adding a
    url-join function similar in capability to ``os.path.join``.

  * Added fixed tile cropping to image transform tool.

  * Added utility functions to detect mimetypes of files via ``file-magic`` or
    ``tika`` optional dependencies.

Web

  * Updated/Rearchitected IqrSearchApp (now IqrSearchDispatcher) to be able to
    spawn multiple IQR configurations during runtime in addition to any
    configured in the input configuration JSON file.  This allows external
    applications to manage configuration storage and generation.

  * Added directory for Girder plugins and added an initial one that, given
    a folder with the correct metadata attached, can initialize an IQR
    instance based on that configuration, and then link to IQR web interface
    (uses existing/updated IqrSearch webapp).

  * Added ability to automatically login via a valid Girder token and parent
    Girder URL for token/user verification. This primarilly allows restricted
    external IQR instance creation and automatic login from Girder redirects.

  * Mongo session information block at bottom IQR app page now only shows up
    when running server in debug mode.

  * Added document showing complete use case with IQR RESTful webservice using
    the IQR docker image with LEEDS Butterfly data. Includes expected results
    users should be able to replicate.


Fixes
-----

Scripts

  * Fixed IQR web app url prefix check

Utilities

  * ``SmqtkObject`` logger class accessor name changed to not conflict with
    ``flask.Flask`` logger instance attribute.

Web

  * Fixed Flow upload browse button to not only allow directory selection
    on Chrome.
