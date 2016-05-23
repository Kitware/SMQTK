Changes Since Last Release
==========================


Updates / New Features since v0.4.0
-----------------------------------

Classifier

  * Added classifier that applies a list of text labels from file to vector
    from descriptor as if it were the classification confidence values.

Descriptor Generators

  * Added ``input_scale`` pass-through option in the Caffe wrapper
    implementation.

  * Added default descriptor factory to yield in-memory descriptors unless
    otherwise instructed.

Descriptor Index

  * Added warning logging message when PostgreSQL implementation file fails to
    import the required python module.

libSVM

  * Tweaked some default parameters in grid.py

LSH Functors

  * Added descriptor normalization option to ITQ functor class.

Scripts

  * Added new output features to classifier model validation script: confusion
    matrix and ROC/PR confidence interval.

  * Moved async batch computation scripts for descriptors, hash codes and
    classifications to ``bin/``.

  * Added script to transform a descriptor index (or part of one) into the
    file format that libSVM likes: ``descriptors_to_svmtrainfile.py``

  * Added script to distort a given image in multiple configurable ways
    including cropping and brigntness/contrast transformations.

  * Added custom scripts resulting from MEMEX April 2016 hackathon.

  * Changed MEMEX update script to collect source ES entries based on crawl
    time instead of insertion time.

Utilities

  * Added async functionality to kernel building functions


Fixes since v0.4.0
------------------

CMake

  * Removed ``SMQTK_FIRST_PASS_COMPLETE`` stuff in root CMakeLists.txt

Scripts

  * Changed ``createFileIngest.py`` so that all specified data elements are
    added to the configured data set at the same time instead of many
    additions.
