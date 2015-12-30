Algorithm Models and Generation
===============================

Some algorithms require a model, of a pre-existing computed state, to function correctly.
Not all algorithm interfaces require that there is some model generation method as it is as times not appropriate or applicable to the definition of the algorithm the interface is for.
However some implementations of algorithms desire a model for some or all of its functionality.
Algorithm implementations that require extra modeling are responsible for providing a method or utility for generating algorithm specific models.
Some algorithm implementations may also be pre-packaged with one or more specific models to optionally choose from, due to some performance, tuning or feasibility constraint.
Explanations about whether an extra model is required and how it is constructed should be detailed by the documentation for that specific implementation.

For example, part of the definition of a ``NearestNeighborsIndex`` algorithm is that there is an index to search over, which is arguably a model for that algorithm.
Thus, the ``build_index()`` method, which should build the index model, is part of that algorithm's interface.
Other algorithms, like the ``DescriptorGenerator`` class of algorithms, do not have a high-level model building method, and model generation or choice is left to specific implementations to explain or provide.

DescriptorGenerator Models
--------------------------

The ``DescriptorGenerator`` interface does not define a model building method, but some implementations require internal models.
Below are explanations on how to build or get modes for ``DescriptorGenerator`` implementations that require a model.

ColorDescriptor
^^^^^^^^^^^^^^^

ColorDescriptor implementations need to build a visual bag-of-words codebook model for reducing the dimensionality of the many low-level descriptors detected in an input data element.
Model parameters as well as storage location parameters are specified at instance construction time, or via a configuration dictionary given to the ``from_config`` class method.

The storage location parameters include a data model directory path and an intermediate data working directory path: ``model_directory`` and ``work_directory`` respectively.
The ``model_directory`` should be the path to a directory for storage of generated model elements.
The ``work_directory`` should be the path to a directory to store cached intermediate data.
If model elements already exist in the provided ``model_directory``, they are loaded at construction time.
Otherwise, the provided directory is used to store model components when the ``generate_model`` method is called.
Please reference the constructor's doc-string for the description of other constructor parameters.

The method ``generate_model(data_set)`` is provided on instances, which should be given an iterable of ``DataElement`` instances representing media that should be used for training the visual bag-of-words codebook.
Media content types that are supported by ``DescriptorGenerator`` instances is listed via the ``valid_content_types()`` method.

Below is an example code snippet of how to train a ColorDescriptor model for some instance of a ColorDescriptor implementation class and configuration:

.. code-block:: python

    # Fill in "<flavor>" with a specific ColorDescriptor class.
    cd = ColorDescriptor_<flavor>(model_directory="data", work_directory="work")

    # Assuming there is not model generated, the following call would fail due to
    # there not being a model loaded
    # cd.compute_descriptor(some_data, some_factory)

    data_elements = [...]  # Some iterable of DataElement instances to media content
    # Generates model components
    cd.generate_model(data_elements)

    # Example of a new instance, given the same parameters, that will load the
    # existing model files in the provided ``model_directory``.
    new_cd = ColorDescriptor_<flavor>(model_directory="data", work_directory="work")

    # Since there is a model, we can now compute descriptors for new data
    new_cd.compute_descriptor(new_data, some_factory)

CaffeDefaultImageNet
^^^^^^^^^^^^^^^^^^^^
This implementation does not come with a method of training its own models, but requires model files provided by Caffe:
the network model file and the image mean binary protobuf file.

The Caffe source tree provides two scripts to download the specific files (relative to the caffe source tree):

.. code-block:: bash

    # Downloads the network model file
    scripts/download_model_binary.py models/bvlc_reference_caffenet

    # Downloads the ImageNet mean image binary protobuf file
    data/ilsvrc12/get_ilsvrc_aux.sh

These script effectively just download files from a specific source.

If the Caffe source tree is not available, the model files can be downloaded from the following URLs:

    - Network model: http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
    - Image mean: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

NearestNeighborsIndex Models (k nearest-neighbors)
--------------------------------------------------

``NearestNeighborsIndex`` interfaced classes include a ``build_index`` method on instances that should build the index model for an implementation.
Implementations, if they allow for persistant storage, should take relevant parameters at construction time.
Currently, we do not package an implementation that require additional model creation.

The general pattern for ``NearestNeighborsIndex`` instance index model generation:

.. code-block:: python

    descriptors = [...]  # some number of descriptors to index

    index = NearestNeighborsIndexImpl(...)
    # Calling ``nn`` should fail before an index has been built.

    index.build_index(descriptors)

    q = DescriptorElementImpl(...)
    neighbors, dists = index.nn(q)

RelevancyIndex Models
---------------------

``RelevancyIndex`` interfaced classes include a ``build_index`` method in instances that should build the index model for a particular implementation.
Implementations, if they allow for persistant storage, should take relevant parameters at construction time.
Currently, we do not package an implementation that requires additional model creation.

The general pattern for ``RelevancyIndex`` instance index model generation:

.. code-block:: python

    descriptors = [...]  # some number of descriptors to index

    index = RelevancyIndexImpl(...)
    # Calling ``rank`` should fail before an index has been built.

    index.build_index(descriptors)

    rank_map = index.rank(pos_descriptors, neg_descriptors)
