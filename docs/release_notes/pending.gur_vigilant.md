SMQTK Vigilant Release Notes
============================


Updates / New Features
----------------------

Algorithms

  Descriptor Generators

    * Added KWCNN DescriptorGenerator plugin

Representation

  Data Elements

    * Added plugin for Girder-hosted data elements

Scripts

  * Added script to add GirderDataElements to a data set

Utilities

  * Started a module containing URL-base utility functions

Web

  * Updated/Rearchitected IqrSearchApp (now IqrSearchDispatcher) to be able to
    spawn multiple IQR configurations during runtime in addition to any
    configured in the input configuration JSON file.  This allows external
    applications to manage configuration storage and generation.

  * Added directory for Girder plugins and added an initial one that, given
    a folder with the correct metadata attached, can initialize an IQR
    instance based on that configuration, and then link to IQR web interface.


Fixes
-----

Scripts

  * Fixed IQR web app url prefix check

Web

  * Fixed Flow upload browse button to not only allow directory selection
    on Chrome.
