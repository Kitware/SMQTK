# SMQTK
![Build Status](https://travis-ci.org/Kitware/SMQTK.svg?branch=master)

Social Multimedia Query ToolKit aims to provide a simple and easy to use interface for content descriptor generation for machine learning, content similarity computation (kNN implementations), and relevancy ranking for online Iterative Query Refinement (IQR) adjudications.

## Intent
SMQTK is intended to be a python toolkit that allows users to, as easily as possibly, apply various machine learning algorithms and techniques to different types of data for different high level applications.
Examples of high level applications may include wanting to search a media corpus for similar content, or providing a relevancy feedback interface for a web application.

## Documentation
Additional documentation can be found in the ``docs`` directory as follows:

* [Architecture Overview](docs/architecture.md)
* [Building/Installing SMQTK](docs/building.md)
* [Data Abstraction/Representation Interfaces](docs/data_representation.md)
* [Algorithm Model Generation](docs/model_generation.md)
* [Example Use Cases](docs/examples.md)


## Basic Descriptor Computation
One of the primary uses for SMQTK is for content descriptor generation.
This section aims to provide a simple example of how to do this for an image file on your local filesystem.

### Setting up the data
Most content description implementations require a model of some kind, and this in turn requires that there be some data corpus to train/generate the model from.
For the purpose of this example, let us assume that this has already been done, and the descriptor specific model is located in a directory called `model`.
See the section `Generating Models` below for more instructions on how to generated content descriptor implementation models, and the section `Forming File-based DataSets` for generating data sets from local files.

### Loading the data
SMQTK uses a data abstraction system in order to mask where data actually exists.
This allows data to be located anywhere from the local file system to data bases to cloud-based services.
For this example, let assume we have a file `foo.png` that we want to compute a descriptor for.
To load the file, we would do the following:

    >>> from smqtk.data_rep.data_element_impl.file_element import DataFileElement
    >>> e = DataFileElement('foo.png')

We can now use `e` as the data source for descriptor computation

### Initializing a Content Descriptor
SMQTK utilizes a general interface for content descriptors and a plugin system to aggregate available implementations at run time.
If the name of the implementation is known, we can use the general class type accessor to get access to available descriptor types.

For the sake of this example, say we want to use the Image CSIFT descriptor as provided by the ColorDescriptor package.
We know, because we looked at the constructor for this class type, that it takes a model directory and a directory to place temporary intermediate working files.

    >>> from smqtk.content_description import get_descriptors
    >>> cd = get_descriptors()['ColorDescriptor_Image_csift']("model", "/tmp")

Content descriptor implementations return a data abstraction of a content descriptor called a `DescriptorElement`.
In order for content descriptor generators to know how to create one, we will need to supply a descriptor element factory instance to the `compute_descriptor` method.
We can create a factory by supplying the desired `DescriptorElement` type and initialization parameters to the generic `DescriptorElementFactory` class.
For this example we will use a file-based descriptor representation, which backs the descriptor vector data to a reproducable file path.
Descriptor element implementations also use a plug-in like framework with a generic accessor method that returns a name-to-class dictionary.

    >>> from smqtk.data_rep import DescriptorElementFactory, get_descriptor_element_impls
    >>> f = DescriptorElementFactory(get_descriptor_element_impls('DescriptorFileElement'),
                                     {"save_dir": "~/descriptor_save_directory"})

We can now compute descriptor vectors for new data, assuming `e` is the `DataFileElement` and `f` is the DescriptorElementFactory` instance created in previous code blocks:

    >>> descr_elem = cd.compute_descriptor(e, f)
    >>> vec = descr_elem.vector()

Where `vec` is a numpy array. The format of this vector is dependent on the descriptor used, but is commonly a 1-dimensional vector of N elements, or an N-dimensional vector, of floats, depending on terminology.


## System Configuration JSON
In the `etc` directory, the `system_config.json` file is intended to provide a central location to map semantic labels to specific configurations of component implementations.
For example, users can map labels to specific data sets, or specific content descriptors trained with specific models, etc.

There is a section for each plugin component system current in SMQTK.
Each section maps strings to sub-dictionaries that specify the implementation type to be associated with the label, and the instance construction parameters to use.
Thus, constructors for implementations, when new ones are made, should only use JSON compliant parameters for non-defaulted constructor arguments.

Comments are allowed in this file when used with SMQTK (`//` line prefix) as they are striped before the JSON is formally parsed (`jsmin` module in `smqtk.utils`).


## Forming File-based DataSets
The lowest level component in SMQTK is the data.
Currently, there are two abstractions: data sets and data elements, where data sets are collections of data elements.

Data elements represent an individual piece of content, like a single image, video, or audio file.
Different implementations of the interface represent different locations where the actual bytes are stored (in-memory, web URL, file, etc.).

Data sets may be created and populated via the interpreter or script.
For file-based data sets, consisting of file-based data elements, a script is provided, `bin/createFileIngest.py`, to conveniently add files to an existing label configuration as configured in the `etc/system_config.json` file.
Other data set implementations may provide a window into a database, or other remote storage services.

Example call to `createFileIngest.py` script that adds all `png` files in a directory to the data set configuration label `example`:

    $> ./bin/createFileIngest.py -s example ./files/*.png

If the number of files to be added to a data set exceeds a shell's maximum argument length, due to glob expansion, the glob should be single-quoted.
This allows the python code to handle the glob, avoiding the maximum argument length issue.

TODO: An abstraction for descriptor vectors has been implemented, but has not been integrated into any other system in SMQTK as of yet.


## Generating Models
Currently, content descriptors and indexer implementations may require a model in order to function.

Usually, these models are based on a training data set, and an indexer is tied to the content descriptor that provided it descriptor vectors.
A convenience script, `bin/generateModel.py`, is provided that generates a content descriptor's model over a configured data set, and then generates an indexer's model using the the generated descriptors.

    $> bin/generateModel.py -d some_dataset -c some_descriptor [ -i some_indexer ]
