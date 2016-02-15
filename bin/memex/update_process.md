# Data and Product Update Process
This is an informational listing of the process for syncing data and computing products for the new data.

## Sync Image files
Currently using ``rsync`` to get new image files from imagecat.dyndns.org via the ``bin/memex/image_dump.rsync_from_imagecat_dyndns.sh`` script.
This results in a log and a test file listing newly added files.

## Computing Descriptors/Hash Codes
Follow the incremental update example at:
 
    http://smqtk.readthedocs.org/en/latest/examples/nnss_incremental_update/incremental_example.html
    
These steps are encapsulated in the ``run.compute_many_descriptors.sh`` and ``run.compute_hash_codes.sh`` scripts.
    
## Computing Classification results

### 3-way gun classifier
