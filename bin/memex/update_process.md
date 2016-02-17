# Data and Product Update Process
This is an informational listing of the process for syncing data and computing products for the new data.

## Sync Image files
Currently using ``rsync`` to get new image files from imagecat.dyndns.org via the ``bin/memex/image_dump.rsync_from_imagecat_dyndns.sh`` script.
This results in a log and a test file listing newly added files.

## Computing Descriptors/Hash Codes
Follow the incremental update example at:
 
    http://smqtk.readthedocs.org/en/latest/examples/nnss_incremental_update/incremental_example.html
    
These steps are encapsulated in the ``run.compute_many_descriptors.sh`` and ``run.compute_hash_codes.sh`` scripts.
    
## Full-stack
Now, the full update process is encapsulated in the ``run.update.sh`` script.

### 3-way gun classifier
Model files:

    - classifier.svm.weapons_3_class_alexnet.labels
    - classifier.svm.weapons_3_class_alexnet.model

Use the ``run.compute_classifications.sh`` script, which wraps a call to the ``compute_classifications.py`` script but with more tightly defined I/O paths.
See the configuration file ``config.jpl_weapons_v3.compute_classifications.json``.
This will classify descriptors based on a UUIDs list, which is created when executing ``run.compute_hash_codes.sh``.
Classifications are stored in the postgres database, as well as output to a CSV file for transport elsewhere (``image_dump.<ts>.classifications.data.csv``).
