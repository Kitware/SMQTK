NearestNeighborServiceServer Incremental Update Example
=======================================================

Goal and Plan
-------------
In this example, we will show how to initially set up an instance of the :class:`.NearestNeighborServiceServer` web API service class such that it can handle incremental updates to its background data.
We will also show how to perform incremental updates and confirm that the service recognizes this new data.

For this example, we will use the :class:`.LSHNearestNeighborIndex` implementation as it is one that currently supports live-reloading its component model files.
Along with it, we will use the :class:`.ItqFunctor` and :class:`.PostgresDescriptorIndex` implementations as the components of the :class:`.LSHNearestNeighborIndex`.
For simplicity, we will not use a specific :class:`.HashIndex`, which causes a :class:`.LinearHashIndex` to be constructed and used at query time.

All scripts used in this example's proceedure have a command line interface that uses dash options.
Their available options can be listed by giving the ``-h``/``--help`` option.
Additional debug logging can be seen output by providing a ``-d`` or ``-v`` option, depending on the script.

Dependencies
````````````
Due to our use of the :class:`.PostgresDescriptorIndex` in this example, a minimum installed version of PostgreSQL 9.4 is required, as is the ``psycopg2`` python module (``conda`` and ``pip`` installable).
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
   :emphasize-lines: 5,6,7,8,9

This step yields two side effects:

    - Descriptors computed are saved in the configured implementation's persistant storage (a postgres database in our case)
    - A file is generated that mapping input files to their :class:`.DataElement` UUID values, or otherwise known as their SHA1 checksum values (:file:`2a.completed_files.csv` for us).
        - This file will be used later as a convenient way of getting at the UUIDs of descriptors processed for a particular ingest.
        - Other uses of this file for other tasks may include:
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
   :emphasize-lines: 5

This step produces the following side effects:

    - Writes the two file components of the model as configured.
        - We configured the output files:
            - :file:`nnss.itq.256bit.mean_vec.npy`
            - :file:`nnss.itq.256bit.rotation.npy`


.. _`step 2c`:

[2c] Computing Hash Codes
'''''''''''''''''''''''''
For this step we will be using the script :file:`bin/scripts/compute_hash_codes.py` to compute ITQ hash codes for the currently computed descriptors.
We will be using the descriptor index we added to before as well as the :class:`.ItqFunctor` models we trained in the previous step.

This script additionally wants to know the UUIDs of the descriptors to compute hash codes for.
We can use the :file:`2a.completed_files.csv` file computed earlier in `step 2a`_ to get at the UUIDs (SHA1 checksum) values for the computed files.
Remember, as is documented in the :class:`.DescriptorGenerator` interface, descriptor UUIDs are the same as the UUID of the data from which it was generated from, thus we can use this file.

We can conveniently extract these UUIDs with the following commands in script :file:`2c.extract_ingest_uuids.sh`, resulting in the file :file:`2c.uuids_for_processing.txt`:

.. literalinclude:: 2c.extract_ingest_uuids.sh
   :language: bash
   :linenos:
   :emphasize-lines: 5

With this file, we can now complete the configuretion for our computation script:

.. literalinclude:: 2c.config.compute_hash_codes.json
   :language: json
   :linenos:

We are not setting a value for ``hash2uuids_input_filepath`` because this is the first time we are running this script, thus we do not have an existing structure to add to.

We can now move forward and run the computation script:

.. literalinclude:: 2c.run.sh
   :language: bash
   :linenos:
   :emphasize-lines: 5

This step produces the following side effects:

    - Writed the file :file:`nnss.hash2uuids.pickle`
        - This file will be used in configuring the :class:`.LSHNearestNeighborIndex` for the :class:`.NearestNeighborServiceServer`



.. _`step 2d`:

[2d] Starting the :class:`.NearestNeighborServiceServer`
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Normally, the :class:`.NearestNeighborsIndex` would need to be built before it can be used.
However, we have effectively already done this in the preceeding steps, so are instead able to get right to configuring and starting the :class:`.NearestNeighborServiceServer`.
