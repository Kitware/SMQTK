NearestNeighborServiceServer Incremental Update Example
=======================================================

Goal and Plan
-------------
In this example, we will show how to initially set up an instance of the :class:`NearestNeighborServiceServer` web API service class such that it can handle incremental updates to its background data.
We will also show how to perform incremental updates and confirm that the service recognizes this new data.

For this example, we will use the :class:`LSHNearestNeighborIndex` implementation as it is one that currently supports live-reloading its component model files.
Along with it, we will use the :class:`ItqFunctor` and :class:`PostgresDescriptorIndex` implementations as the components of the :class:`LSHNearestNeighborIndex`.
For simplicity, we will not use a specific :class:`HashIndex`, which causes a :class:`LinearHashIndex` to be constructed and used at query time.

All scripts used in this example's proceedure have a command line interface that uses dash options.
Their available options can be listed by giving the ``-h``/``--help`` option.
Additional debug logging can be seen output by providing a ``-d`` or ``-v`` option, depending on the script.

Dependencies
````````````
Due to our use of the :class:`PostgresDescriptorIndex` in this example, a minimum installed version of PostgreSQL 9.4 is required, as is the ``psycopg2`` python module (``conda`` and ``pip`` installable).
Please check and modify the configuration files for this example to be able to connect to the database of your choosing.

Take a look at the :file:`etc/smqtk/postgres/descriptor_element/example_table_init.sql` and :file:`etc/smqtk/postgres/descriptor_index/example_table_init.sql` files, located from the root of the source tree, for table creation examples for element and index storage:

.. code-block:: bash

   $ psql postgres -f etc/smqtk/postgres/descriptor_element/example_table_init.sql
   $ psql postgres -f etc/smqtk/postgres/descriptor_index/example_table_init.sql


Proceedure
----------

.. _`step 1`:

[1] Getting and Splitting the data set
``````````````````````````````````````
For this example we will use the `Leeds butterfly data set`_ (see the :file:`download_leeds_butterfly.sh` script).
We will split the data set into an initial sub-set composed of about half of the images from each butterfly catagory (418 total images in the :file:`2.ingest_files_1.txt` file).
We will then split the data into a two more sub-sets each composed of about half of the remaining data (each composing about 1/4 of the original data set, totaling 209 and 205 images each in the :file:`TODO.ingest_files_2.txt` and :file:`TODO.ingest_files_3.txt` files respectively).

.. _`Leeds butterfly data set`: http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/

.. _`step 2`:

[2] Computing Initial Ingest
````````````````````````````
For this example, an "ingest" consists of a set of descriptors in an index and a mapping of hash codes to the descriptors.

In this example, we also train the LSH hash code functor's model, if it needs one, based on the descriptors computed before computing the hash codes.
We are using the ITQ functor which does require a model.
It may be the case that the functor of choice does not require a model, or a sufficient model for the functor is already available for use, in which case that step may be skipped.

Our example's initial ingest will use the image files listed in the :file:`2.ingest_files_1.txt` test file.

.. _`step 2a`:

[2a] Computing Descriptors
''''''''''''''''''''''''''
We will use the script ``bin/scripts/compute_many_descriptors.py`` for computing descriptors from a list of file paths.
This script will be used again in later sections for additional incremental ingests.

The example configuration file for this script, :file:`2a.config.compute_many_descriptors.json` (shown below), should be modified to connect to the appropriate PostgreSQL database and the correct Caffe model files for your system.
For this example, we will be using Caffe's ``bvlc_alexnet`` network model with the ``ilsvrc12`` image mean be used for this example.

.. literalinclude:: 2a.config.compute_many_descriptors.json
   :language: json
   :linenos:

For running the script, take a look at the example invocation in the file :file:`2a.run.sh`:

.. literalinclude:: 2a.run.sh
   :language: bash
   :linenos:

This step yields two side effects:

    - Descriptors computed are saved in the configured implementation's persistant storage (a postgres database in our case)
    - A file is generated that mapping input files to their UUID values, or otherwise known as their SHA1 checksum values (:file:`2a.completed_files.csv` for us).
        - This file is not needed for the rest of this example, but may be important if:
            - interfacing with other systems that use file paths as the primary identifier of base data files
            - want to quickly back-reference the original file for a given UUID, as UUIDs for descriptor and classification elements are currently the same as the original file they are computed from.

.. _`step 2b`:

[2b] Training ITQ Model
'''''''''''''''''''''''
To train the ITQ model, we will use the script: :file:`./bin/scripts/train_itq.py`.
We'll want to train the functor's model using the descriptors computed in `step 2a`_.
Since we will be using the whole index (418 descriptors), we will not need to provide the script with an additional list of UUIDs.

The example configuration file for this script, :file:`2b.config.train_itq.json`, should be modified to connect to the appropriate PostgreSQL database.

.. literalinclude:: 2b.config.train_itq.json
   :language: json
   :linenos:

:file:`2b.run.sh` contains an example call of the training script:

.. literalinclude:: 2b.run.sh
   :language: bash
   :linenos:

This step produces the following side effects:

    - Writes out the two file components of the model as configured: :file:`itq.256bit.mean_vec.npy` and :file:`itq.256bit.rotation.npy`.


[2c] Computing Hash Codes
'''''''''''''''''''''''''
For this step we will be using the script :file:`bin/scripts/compute_hash_codes.py`.

[2d] Building the LSH NN-Index
''''''''''''''''''''''''''''''
