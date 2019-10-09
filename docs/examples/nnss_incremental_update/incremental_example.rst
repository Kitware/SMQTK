NearestNeighborServiceServer Incremental Update Example
=======================================================

Goal and Plan
-------------
In this example, we will show how to initially set up an instance of the :class:`.NearestNeighborServiceServer` web API service class such that it can handle incremental updates to its background data.
We will also show how to perform incremental updates and confirm that the service recognizes this new data.

For this example, we will use the :class:`.LSHNearestNeighborIndex` implementation as it is one that currently supports live-reloading its component model files.
Along with it, we will use the :class:`.ItqFunctor` and :class:`.PostgresDescriptorSet` implementations as the components of the :class:`.LSHNearestNeighborIndex`.
For simplicity, we will not use a specific :class:`.HashIndex`, which causes a :class:`.LinearHashIndex` to be constructed and used at query time.

All scripts used in this example's proceedure have a command line interface that uses dash options.
Their available options can be listed by giving the ``-h``/``--help`` option.
Additional debug logging can be seen output by providing a ``-d`` or ``-v`` option, depending on the script.

This example assumes that you have a basic understanding of:

    - JSON for configuring files
    - how to use the :file:`bin/runApplication.py`
    - SMQTK's NearestNeighborServiceServer application and algorithmic/data-structure components.
        - :class:`.NearestNeighborsIndex`, specific the implementation :class:`.LSHNearestNeighborIndex`
        - :class:`.DescriptorSet` abstract and implementations with an updatable persistance storage mechanism (e.g. :class:`.PostgresDescriptorSet`).
        - :class:`.LshFunctor` abstract and implementations

Dependencies
````````````
Due to our use of the :class:`.PostgresDescriptorSet` in this example, a minimum installed version of PostgreSQL 9.4 is required, as is the ``psycopg2`` python module (``conda`` and ``pip`` installable).
Please check and modify the configuration files for this example to be able to connect to the database of your choosing.

Take a look at the :file:`etc/smqtk/postgres/descriptor_element/example_table_init.sql` and :file:`etc/smqtk/postgres/descriptor_set/example_table_init.sql` files, located from the root of the source tree, for table creation examples for element and index storage:

.. code-block:: bash

   $ psql postgres -f etc/smqtk/postgres/descriptor_element/example_table_init.sql
   $ psql postgres -f etc/smqtk/postgres/descriptor_set/example_table_init.sql


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
            - :file:`2b.itq.256bit.mean_vec.npy`
            - :file:`2b.nnss.itq.256bit.rotation.npy`


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

With this file, we can now complete the configuration for our computation script:

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

    - Writed the file :file:`2c.hash2uuids.pickle`
        - This file will be copied and used in configuring the :class:`.LSHNearestNeighborIndex` for the :class:`.NearestNeighborServiceServer`


.. _`step 2d`:

[2d] Starting the :class:`.NearestNeighborServiceServer`
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Normally, a :class:`.NearestNeighborsIndex` instance would need to be have its index built before it can be used.
However, we have effectively already done this in the preceeding steps, so are instead able to get right to configuring and starting the :class:`.NearestNeighborServiceServer`.
A default configuration may be generated using the generic :file:`bin/runApplication.py` script (since web applications/servers are plugins) using the command::

    $ runApplication.py -a NearestNeighborServiceServer -g 2d.config.nnss_app.json

An example configuration has been provided in :file:`2d.config.nnss_app.json`.
The :class:`.DescriptorSet`, :class:`.DescriptorGenerator` and :class:`.LshFunctor` configuration sections should be the same as used in the preceeding sections.

Before configuring, we are copying :file:`2c.hash2uuids.pickle` to :file:`2d.hash2uuids.pickle`.
Since we will be overwriting this file (the ``2d`` version) in steps to come, we want to separate it from the results of `step 2c`_.

Note the highlighted lines for configurations of note for the :class:`.LSHNearestNeighborIndex` implementation.
These will be explained below.

.. literalinclude:: 2d.config.nnss_app.json
   :language: json
   :linenos:
   :emphasize-lines: 55,56,58,61,72

Emphasized line explanations:

    - On line ``55``, we are using the ``hik`` distance method, or histogram intersection distance, as it has been experimentally shown to out perform other distance metrics for AlexNet descriptors.
    - On line ``56``, we are using the output generated during `step 2c`_.
      This file will be updated during incremental updates, along with the configured :class:`.DescriptorSet`.
    - On line ``58``, we are choosing not to use a pre-computed :class:`.HashIndex`.
      This means that a :class:`.LinearHashIndex` will be created and used at query time.
      Other implementations in the future may incorporate live-reload functionality.
    - On line ``61``, we are telling the :class:`.LSHNearestNeighborIndex` to reload its implementation-specific model files when it detects that they've changed.
        - We listed :class:`.LSHNearestNeighborIndex` implementation's only model file on line ``56`` and will be updated via the :file:`bin/scripts/compute_hash_codes.py`
    - On line ``72``, we are telling the implementation to make sure it does not write to any of its resources.

We can now start the service using::

    $ runApplication.py -a NearestNeighborServiceServer -c 2d.config.nnss_app.json

We can test the server by calling its web api via curl using one of our ingested images, :file:`leedsbutterfly/images/001_0001.jpg`::

    $ curl http://127.0.0.1:5000/nn/n=10/file:///home/purg/data/smqtk/leedsbutterfly/images/001_0001.jpg
    {
      "distances": [
        -2440.0882132202387,
        -1900.5749250203371,
        -1825.7734497860074,
        -1771.708476960659,
        -1753.6621350347996,
        -1729.6928340941668,
        -1684.2977819740772,
        -1627.438737615943,
        -1608.4607088603079,
        -1536.5930510759354
      ],
      "message": "execution nominal",
      "neighbors": [
        "84f62ef716fb73586231016ec64cfeed82305bba",
        "ad4af38cf36467f46a3d698c1720f927ff729ed7",
        "2dffc1798596bc8be7f0af8629208c28606bba65",
        "8f5b4541f1993a7c69892844e568642247e4acf2",
        "e1e5f3e21d8e3312a4c59371f3ad8c49a619bbca",
        "e8627a1a3a5a55727fe76848ba980c989bcef103",
        "750e88705efeee2f12193b45fb34ec10565699f9",
        "e21b695a99fee6ff5af8d2b86d4c3e8fe3295575",
        "0af474b31fc8002fa9b9a2324617227069649f43",
        "7da0501f7d6322aef0323c34002d37a986a3bf74"
      ],
      "reference_uri": "file:///home/purg/data/smqtk/leedsbutterfly/images/001_0001.jpg",
      "success": true
    }

If we compare the result neighbor UUIDs to the SHA1 hash signatures of the original files (that descritpors were computed from), listed in the `step 2a`_ result file :file:`2a.completed_files.csv`, we find that the above results are all of the class ``001``, or monarch butterflies.

If we used either of the files :file:`leedsbutterfly/images/001_0042.jpg` or :file:`leedsbutterfly/images/001_0063.jpg`, which are not in our initial ingest, but in the subsequent ingests, and set ``.../n=832/...`` (the maximum size we will see in ingest grow to), we would see that the API does not return their UUIDs since they have not been ingested yet.
We will also see that only 418 neighbors are returned even though we asked for 832, since there are only 418 elements currently in the index.
We will use these three files as proof that we are actually expanding the searchable content after each incremental ingest.

We provide a helper bash script, :file:`test_in_index.sh`, for checking if a file is findable via in the search API.
A call of the form::

    $ ./test_in_index.sh leedsbutterfly/images/001_0001.jpg 832

... performs a curl call to the server's default host address and port for the 832 nearest neighbors to the query image file, and checks if the UUIDs of the given file (the sha1sum) is in the returned list of UUIDs.


[3] First Incremental Update
````````````````````````````
Now that we have a live :class:`.NearestNeighborServiceServer` instance running, we can incrementally process the files listed in :file:`3.ingest_files_2.txt`, making them available for search without having to shut down or otherwise do anything to the running server instance.

.. _2a: `step 2a`_
.. _2c: `step 2c`_

We will be performing the same actions taken in steps 2a_ and 2c_, but with different inputs and outputs:

  1. Compute descriptors for files listed in :file:`3.ingest_files_2.txt` using script :file:`compute_many_descriptors.py`, outputting file :file:`3.completed_files.csv`.
  2. Create a list of descriptor UUIDs just computed (see :file:`2c.extract_ingest_uuids.sh`) and compute hash codes for those descriptors, overwriting :file:`2d.hash2uuids.pickle` (which causes the server the :class:`.LSHNearestNeighborIndex` instance to update itself).

The following is the updated configuration file for hash code generation. Note the highlighted lines for differences from `step 2c`_ (notes to follow):

.. literalinclude:: 3.config.compute_hash_codes.json
   :language: json
   :linenos:
   :emphasize-lines: 31,32,36

Line notes:

    - Lines ``31`` and ``32`` are set to the model file that the :class:`.LSHNearestNeighborIndex` implementation for the server was configured to use.
    - Line ``36`` should be set to the descriptor UUIDs file generated from :file:`3.completed_files.csv` (see :file:`2c.extract_ingest_uuids.sh`)

The provided :file:`3.run.sh` script is an example of the commands to run for updating the indices and models:

.. literalinclude:: 3.run.sh
   :language: bash
   :linenos:

After calling the :file:`compute_hash_codes.py` script, the server logging should yield messages (if run in debug/verbose mode) showing that the :class:`.LSHNearestNeighborIndex` updated its model.

We can now test that the :class:`.NearestNeighborServiceServer` using the query examples used at the end of `step 2d`_.
Using images :file:`leedsbutterfly/images/001_0001.jpg` and :file:`leedsbutterfly/images/001_0042.jpg` as our query examples (and ``.../n=832/...``), we can see that both are in the index (each image is the nearest neighbor to itself).
We also see that a total of 627 neighbors are returned, which is the current number of elements now in the index after this update.
The sha1 of the third image file, :file:`leedsbutterfly/images/001_0082.jpg`, when used as the query example, is not included in the returned neighbors and thus found in the index.


[4] Second Incremental Update
`````````````````````````````
Let us repeat again the above process, but using the third increment set (highlighted lines different from :file:`3.run.sh`):

.. literalinclude:: 4.run.sh
  :language: bash
  :linenos:
  :emphasize-lines: 11,12,15,19

After this, we should be able to query all three example files used before and see that they are all now included in the index.
We will now also see that all 832 neighbors requested are returned for each of the queries, which equals the total number of files we have ingested over the above steps.
If we increase ``n`` for a query, only 832 neighbors are returned, showing that there are 832 elements in the index at this point.
