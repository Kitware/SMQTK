# SMQTK Architecture Overview

SMQTK is mainly comprised of 3 components

* Data abstraction interfaces and plugin implementations
* Algorithm abstraction interfaces and plugin implementations
* Web services and script utilities that used both of the above


## Data Abstraction
An important part of any algorithm is the data its working over and the data that it produces.
An important part of working with large scales of data is where the data is stored and how its accessed.
The ``smqtk.representation`` module contains interfaces and plugins for various core data structures, allowing plugin implementations to decide where and how the underlying raw data should be stored and accessed.
This potentially allows algorithms to handle more data that would otherwise be feasible on a single machine.


### Data Representation Structures
The following are the core data representation interfaces.
Each bullet links to the source file in which the interface is defined, which contains more documentation on the intent and function of its high level functionality.

* [``CodeIndex``](python/smqtk/representation/code_index/__init__.py)
    * Mapping structure from a bit-code (represented as a python ``integer`` or ``long``) to one or more associated ``DescriptorElement`` instances.
* [``DataElement``](python/smqtk/representation/data_element/__init__.py)
    * High level interface to media content.
* [``DataSet``](python/smqtk/representation/data_set/__init__.py)
    * Set-based container for ``DataElement`` instances.
* [``DescriptorElement``](python/smqtk/representation/descriptor_element/__init__.py)
    * Container for an arbitrary descriptor vector, or more simply, a vector of floating point values.

It is required that implementations have a common serialization format so that they may be stored or transported by other structures in a general way without caring what the specific implementation is.
For this we require that all implementations be serializable via the ``pickle`` (and thus ``cPickle``) module functions.


### Data Support Structures
Other data structures are provided in the ``smqtk.representation`` module to assist with the use of the above described structures:

* [``DescriptorElementFactory``](python/smqtk/representation/descriptor_element_factory.py)
    * Factory object for producing ``DescriptorElement`` instances given a type and uuid.
    * Used for when something knows it wants to produce ``DescriptorElement`` instances without caring what specific implementation of ``DescriptorElement`` is being produced.


## Algorithm Interfaces


## Utilities and Applications
