Data Abstraction
----------------

An important part of any algorithm is the data its working over and the data that it produces.
An important part of working with large scales of data is where the data is stored and how its accessed.
The ``smqtk.representation`` module contains interfaces and plugins for various core data structures, allowing plugin implementations to decide where and how the underlying raw data should be stored and accessed.
This potentially allows algorithms to handle more data that would otherwise be feasible on a single machine.

.. autoclass:: smqtk.representation.SmqtkRepresentation


Data Representation Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following are the core data representation interfaces.

CodeIndex
+++++++++

.. automodule:: smqtk.representation.code_index
   :members:
.. autoclass:: smqtk.representation.code_index.CodeIndex
   :members:

DataElement
+++++++++++

.. automodule:: smqtk.representation.data_element
   :members:
.. autoclass:: smqtk.representation.data_element.DataElement
   :members:

DataSet
+++++++

.. automodule:: smqtk.representation.data_set
   :members:
.. autoclass:: smqtk.representation.data_set.Dataset
   :members:

DescriptorElement
+++++++++++++++++

.. automodule:: smqtk.representation.descriptor_element
   :members:
.. autoclass:: smqtk.representation.descriptor_element.DescriptorElement
   :members:

It is required that implementations have a common serialization format so that they may be stored or transported by other structures in a general way without caring what the specific implementation is.
For this we require that all implementations be serializable via the ``pickle`` (and thus ``cPickle``) module functions.


Data Support Structures
^^^^^^^^^^^^^^^^^^^^^^^

Other data structures are provided in the [``smqtk.representation``](/python/smqtk/representation) module to assist with the use of the above described structures:

DescriptorElementFactory
++++++++++++++++++++++++

.. automodule:: smqtk.representation.descriptor_element_factory
   :members:
.. autoclass:: smqtk.representation.descriptor_element_factory.DescriptorElementFactory
   :members:
