# Algorithm Models and Generation
Some algorithms require a model, of a pre-existing computed state, to function correctly.
Not all algorithm interfaces require that there is some model generation method as it is as times not appropriate or applicable to the definition of the algorithm the interface is for.
However some implementations of algorithms desire a model for some or all of its functionality.
Algorithm implementations that require extra modeling are responsible for providing a method or utility for generating algorithm specific models.
Some algorithm implementations may also be pre-packaged with one or more specific models to optionally choose from, due to some performance, tuning or feasibility constraint.
Explanations about whether an extra model is required and how it is constructed should be detailed by the documentation for that specific implementation.

For example, part of the definition of a ``NearestNeighborsIndex`` algorithm is that there is an index to search over, which is arguably a model for that algorithm.
Thus, the ``build_index()`` method, which should build the index model, is part of that algorithm's interface.
Other algorithms, like the ``DescriptorGenerator`` class of algorithms, do not have a high-level model building method, and model generation or choice is left to specific implementations to explain or provide.

## DescriptorGenerator Models
The ``DescriptorGenerator`` interface does not define a model building method, but some implementations require internal models.
Below are explanations on how to build or get modes for ``DescriptorGenerator`` implementations that require a model.

### ColorDescriptor
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

Below is an example code snippet of how to train a ColorDescriptor model for some instance of a ColorDescriptor implementation class and configuration.

```python
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
```

Here is a concrete example of performing this for a set of ten butterfly images:

```python
# Import some butterfly data
urls = ["http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/examples/{:03d}.jpg".format(i) for i in range(1,11)]
from smqtk.representation.data_element.url_element import DataUrlElement
el = [DataUrlElement(d) for d in urls]

# Create a model. This assumes you have set up the colordescriptor executable.
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
cd = get_descriptor_generator_impls()['ColorDescriptor_Image_csift'](model_directory='data', work_directory='work')
cd.generate_model(el)

# Set up a factory for our vector (here in-memory storage)
from smqtk.representation.descriptor_element_factory import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
factory = DescriptorElementFactory(DescriptorMemoryElement, {})

# Compute features on the first image
result = cd.compute_descriptor(el[0], factory)
result.vector()
# array([ 0.        ,  0.01254855,  0.        , ...,  0.0035853 ,
#         0.        ,  0.00388408])
```

## NearestNeighborsIndex Models (k nearest-neighbors)
``NearestNeighborsIndex`` interfaced classes include a ``build_index`` method on instances that should build the index model for an implementation.
Implementations, if they allow for persistant storage, should take relevant parameters at construction time.
Currently, we do not package an implementation that require additional model creation.

The general pattern for ``NearestNeighborsIndex`` instance index model generation:

```python
descriptors = [...]  # some number of descriptors to index

index = NearestNeighborsIndexImpl(...)
# Calling ``nn`` should fail before an index has been built.

index.build_index(descriptors)

q = DescriptorElementImpl(...)
neighbors, dists = index.nn(q)
```

## RelevancyIndex Models
``RelevancyIndex`` interfaced classes include a ``build_index`` method in instances that should build the index model for a particular implementation.
Implementations, if they allow for persistant storage, should take relevant parameters at construction time.
Currently, we do not package an implementation that requires additional model creation.

The general pattern for ``RelevancyIndex`` instance index model generation:

```python
descriptors = [...]  # some number of descriptors to index

index = RelevancyIndexImpl(...)
# Calling ``rank`` should fail before an index has been built.

index.build_index(descriptors)

rank_map = index.rank(pos_descriptors, neg_descriptors)
```
