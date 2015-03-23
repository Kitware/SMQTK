# Dependencies
In order to provide complete functionality, the following dependencies are required:

* ColorDescriptor
  * For CSIFT, TCH, etc. feature descriptors.
  * http://koen.me/research/colordescriptors/
  * After unpacking the downloaded ZIP archive, add the directory it was
    extracted to to the PYTHONPATH so the DescriptorIO.py module can be
    accessed and used within the SMQTK library.
* MongoDB
  * MongoDB is required for the Web application for session storage. This
    allows the Flask application API to modify session contents when within
    AJAX routines. which is sometimes required for asynchronous user state
    interaction/modification.
  * This is not a perminent requirement as other mediums can be used for this
    purpose, however they would need implementation.
