
Utilities and Applications
--------------------------

Also part of SMQTK are support utility modules, utility scripts (effectively the "binaries") and service-oriented and demonstration web applications.

Utility Modules
^^^^^^^^^^^^^^^

Various unclassified functionality intended to support the primary goals of SMQTK.
See doc-string comments on sub-module classes and functions in [``smqtk.utils``](/python/smqtk/utils) module.

Utility Scripts
^^^^^^^^^^^^^^^

Located in the [``smqtk.bin``](/python/smqtk/bin) module are various scripts intended to provide quick access or generic entry points to common SMQTK functionality.
These scripts generally require configuration via a JSON text file and executable entry points are installed via the ``setup.py``.
By rule of thumb, scripts that require a configuration also provide an option for outputting a default or example configuration file.

Currently available utility scripts in alphabetical order:

classifier_kfold_validation
+++++++++++++++++++++++++++
.. argparse::
   :ref: smqtk.bin.classifier_kfold_validation.cli_parser
   :prog: classifier_kfold_validation

classifier_model_validation
+++++++++++++++++++++++++++
.. argparse::
   :ref: smqtk.bin.classifier_model_validation.cli_parser
   :prog: classifier_model_validation

classifyFiles
+++++++++++++
.. argparse::
   :ref: smqtk.bin.classifyFiles.get_cli_parser
   :prog: classifyFiles

compute_classifications
+++++++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.compute_classifications.cli_parser
    :prog: compute_classifications

compute_hash_codes
++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.compute_hash_codes.cli_parser
    :prog: compute_hash_codes

compute_many_descriptors
++++++++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.compute_many_descriptors.cli_parser
    :prog: compute_many_descriptors

computeDescriptor
+++++++++++++++++
.. argparse::
   :ref: smqtk.bin.computeDescriptor.cli_parser
   :prog: computeDescriptor

createFileIngest
++++++++++++++++
.. argparse::
   :ref: smqtk.bin.createFileIngest.cli_parser
   :prog: createFileIngest

descriptors_to_svmtrainfile
+++++++++++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.descriptors_to_svmtrainfile.cli_parser
    :prog: descriptors_to_svmtrainfile

generate_image_transform
++++++++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.generate_image_transform.cli_parser
    :prog: generate_image_transform

iqr_app_model_generation
++++++++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.iqr_app_model_generation.cli_parser
    :prog: iqr_app_model_generation

iqrTrainClassifier
++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.iqrTrainClassifier.get_cli_parser
    :prog: iqrTrainClassifier

make_balltree
+++++++++++++
.. argparse::
    :ref: smqtk.bin.make_balltree.cli_parser
    :prog: make_balltree

minibatch_kmeans_clusters
+++++++++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.minibatch_kmeans_clusters.cli_parser
    :prog: minibatch_kmeans_clusters

proxyManagerServer
++++++++++++++++++
.. argparse::
    :ref: smqtk.bin.proxyManagerServer.cli_parser
    :prog: proxyManagerServer

removeOldFiles
++++++++++++++
.. argparse::
   :ref: smqtk.bin.removeOldFiles.cli_parser
   :prog: removeOldFiles

runApplication
++++++++++++++
Generic entry point for running SMQTK web applications defined in [``smqtk.web``](/python/smqtk/web).

.. argparse::
    :ref: smqtk.bin.runApplication.cli_parser
    :prog: runApplication

summarizePlugins
++++++++++++++++
.. argparse::
    :ref: smqtk.bin.summarizePlugins.cli
    :prog: summarizePlugins

train_itq
+++++++++
.. argparse::
    :ref: smqtk.bin.train_itq.cli_parser
    :prog: train_itq
