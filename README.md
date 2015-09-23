# SMQTK
![Build Status](https://travis-ci.org/Kitware/SMQTK.svg?branch=master)

Social Multimedia Query ToolKit aims to provide a simple and easy to use interface for content descriptor generation for machine learning, content similarity computation (kNN implementations), and relevancy ranking for online Iterative Query Refinement (IQR) adjudications.


## Dependencies
In order to provide basic functionality:

* Build SMQTK via CMAKE.
  * Currently, a naive CMake configuration (no modifications to options) is acceptable for basic functionality.
* Install python packages detailed in the `requirements.*.txt` files.

In order to run provided SMQTKSearchApp web application, the following are additionally required:

* MongoDB
  * MongoDB is required for the Web application for session storage.
    This allows the Flask application API to modify session contents when within AJAX routines.
    This required for asynchronous user-session state interaction/modification.
  * This is not a permanent requirement as other mediums can be used for this purpose, however they would need implementation.

### Installing Python dependencies
There are two files that list required python packages:

* requirements.conda.txt
* requirements.pip.txt

Required packages have been split up this way because conda does not provide all packages that pip can.
While conda is generally considered the preferred method of acquiring python dependencies due to their pre-built nature, some of our requirements are not available through conda.

#### Installing with Conda and Pip
The three-step python dependency installation using both conda and pip will look like the following:

    $ conda create -n <env_name> --file requirements.conda.txt
    $ . activate <env_name>
    $ pip install -r requirements.pip.txt

#### Installing with just Pip
If installation of python dependencies via pip only is desired, or if local compilation of packages is desired, the following is recommended:

    $ pip install -r requirements.conda.txt -r requirements.pip.txt

##### NumPy and SciPy
If installing NumPy and SciPy via pip, it may be useful or required to install BLAS or LAPACK libraries for certain functionality and efficiency.

### Additional Descriptor Dependencies
Descriptors implemented in SMQTK may require additional dependencies in order to run.
This may be because a descriptor required additional libraries or tools on the system.
For example, the ColorDescriptor implementation required a 3rd party tool to downloaded and setup.

* ColorDescriptor
  * For CSIFT, TCH, etc. feature descriptors.
  * http://koen.me/research/colordescriptors/
  * After unpacking the downloaded ZIP archive, add the directory it was extracted to to the PYTHONPATH so the DescriptorIO.py module can be accessed and used within the SMQTK library.
  * Note that a license is required for commercial use (See the koen.me webpage).

As more descriptors are added, more optional dependencies may be introduced.


## Building
Building SMQTK requires CMake and a C/C++ compiler.
See the example below for a simple example of how to build SMQTK

### CMake Build
Navigate to where the build products should be located.
It is recommended that this not be the source tree.
Build products include some C/C++ libraries, python modules and generated scripts.

If the desired build directory, and run the following, filling in ``<...>`` with appropriate values:

    $ cmake <source_dir_path>

Optionally, the `ccmake` command line utility, or the GUI version, may be run in order to modify options for building additional modules.
Currently, the selection is very minimal, but may be expanded over time.

### Example
```bash
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
. build/setup_smqtk.build.sh
# Running tests
source/run_tests.sh
```


## Algorithm Models
Some algorithms require a model, of a pre-existing computed state, to function correctly.
Not all algorithm interfaces require that there is some model generation method as it is as times not appropriate or applicable to the definition of the algorithm the interface is for.
However some implementations of algorithms desire a model for some or all of its functionality.
Algorithm implementations that require extra modeling are responsible for providing a method or utility for generating algorithm specific models.
Some algorithm implementations may also be pre-packaged with one or more specific models to optionally choose from, due to some performance, tuning or feasibility constraint.
Explanations about whether an extra model is required and how it is constructed should be detailed by the documentation for that specific implementation.

For example, part of the definition of a ``NearestNeighborsIndex`` algorithm is that there is an index to search over, which is arguably a model for that algorithm.
Thus, the ``build_index()`` method, which should build the index model, is part of that algorithm's interface.
Other algorithms, like the ``ContentDescriptor`` class of algorithms, do not have a high-level model building method, and model generation or choice is left to specific implementations to explain or provide.

### ContentDescriptor Models
The ``ContentDescriptor`` interface does not define a model building method, but some implementations require internal models.
Below is explanations on how to build modes for ``ContentDescriptor`` implementations that require a model.

#### ColorDescriptor
ColorDescriptor implementations need to build a visual bag-of-words codebook model for reducing the dimensionality of the many low-level descriptors detected in an input data element.
Model parameters as well as storage location parameters are specified at instance construction time, or via a configuration dictionary given to the ``from_config`` class method.

The storage location parameters include a data model directory path and an intermediate data working directory path: ``model_directory`` and ``work_directory`` respectively.
The ``model_directory`` should be the path to a directory for storage of generated model elements.
The ``work_directory`` should be the path to a directory to store cached intermediate data.
If model elements already exist in the provided ``model_directory``, they are loaded at construction time.
Otherwise, the provided directory is used to store model components when the ``generate_model`` method is called.
Please reference the constructor's doc-string for the description of other constructor parameters.

The method ``generate_model(data_set)`` is provided on instances, which should be given an iterable of ``DataElement`` instances representing media that should be used for training the visual bag-of-words codebook.
Media content types that are supported by ``ContentDescriptor`` instances is listed via the ``valid_content_types()`` method.

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

### NearestNeighborsIndex Models (k nearest-neighbors)
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

### RelevancyIndex Models
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
