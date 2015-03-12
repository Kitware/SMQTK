TODO: Formalize this. For now its mostly a smattering of information


Ingests and FeatureDescriptor data sets
---------------------------------------
Feature Descriptor data sets are made in respect to a particular ingest. Due
to this, IDs of files in the ingest should correspond 1-to-1 with the feature
data IDs in a data set.

This is how an IqrSession knows how to set the ID of extension images even
though it doesn't have explicit access to the IngestManager object associated
the feature descriptor data it has.


Ingest / Model Creation
-----------------------
The following scripts located in the bin directory are provided for convenient
ingest creation:

    - ./bin/ingest_images.py
    - ./bin/generate_data_models_from_ingest.py

First, make sure that your build or install environment is set up (i.e. source
the appropriate setup script), and then call the ``ingest_images.py`` script
given shell glob arguments pointing to image files to ingest. This script my be
called multiple times to add to the same ingest (assuming no masir_config
changes in between runs). Ingested data is saved in the DIR_DATA location
specified in the masir_config.py file.

Once all images are accumulated into the ingest, the
``generate_data_models_from_ingest.py`` script may be called, which takes into
consideration all imagery currently in the ingest and creates the data models
required for each descriptor configured in the script

TODO: Modify this script to pick up and use the models.json config file in etc.
      That file probably needs format updates.
