Data Classifier Service
=======================
This service provides the ability to classify arbitrary data using the
configured descriptor generator and available classifier models.


Descriptor Generation
---------------------
One descriptor generator is configured for the whole service, so this limits
the data types that a single service instance will accept as descriptor
generators define what data types they function over. This also constrains the
classifiers that should be configured for a server since configured classifiers
need to be able to classify the generated descriptors.

Descriptor Element Factory
^^^^^^^^^^^^^^^^^^^^^^^^^^
A :ref:`smqtk.representation.DescriptorElementFactory` needs to be configured
in order to tell the generator algorithm where to store descriptors.

Generally, for the purposes of this server, in-memory elements are sufficient
since we are not collecting them as well as they are the simplest and fastest
implementation to interact with.

It may be more appropriate to choose a backend with persistent storage in cases
where the same data is expected to be given to this service many times.


Data Classification
-------------------
This service allows multiple methods for making classifiers available for use
via the REST interface:

    - Statically defined in configuration file.
    - Building classifiers from IQR session state files as saved from the IQR
      RESTful service or IQR GUI web-application.

Configured Classifiers
^^^^^^^^^^^^^^^^^^^^^^
A user-specified number of classifier plugin configurations can be defined in
the configuration file for the service. The ``constant_classifiers`` should be
a mapping of the semantic label of the classifier to the classifier plugin
configuration.

Classifiers from IQR states
^^^^^^^^^^^^^^^^^^^^^^^^^^^
One configuration section defines a supervised classifier configuration to be
used when building classifiers when the REST service is handed IQR state blob.

Classification Element Factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A :ref:`smqtk.representation.ClassificationElementFactory` is configured here
in order to define where classification results are to be stored.

For the purposes of this server, in-memory elements should be since we are
not collecting them as well as they are the simplest and fastest implementation
to interact with.

Using a backend with persistent storage is not recommended due to multiple
classifiers of the same algorithmic type overwriting each-other's stored
results. This occurs due to how classification elements inherit the type name of
the algorithm that generated it, and the UUID of the descriptor that was
classified.
