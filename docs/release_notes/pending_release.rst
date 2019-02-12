SMQTK Pending Release Notes
===========================


Updates / New Features
----------------------

Documentation

    - Updated IQR Demo Application documentation RST file and images to
      reflect the current state of SMQTK and that process.


Fixes
-----

Algorithms

    - Classifiers

        - SVM

            - Fixed broken large model saving in Python 2, creating
              parity with Python 3.

    - Nearest-Neighbors

        - FAISS

            - Fixed use of strings for compatibility with Python 2.
            - Fixed broken large model saving in Python 2, creating
              parity with Python 3.

        - FLANN

            - Fixed broken large model saving in Python 2, creating
              parity with Python 3.

        - Hash Index

            - Scikit-Learn BallTree

                - Fix ``save_model`` and ``load_model`` methods for additional
                  compatibility with scikit-learn version 0.20.0.

        - LSH

            - Fix issue with update and remove methods when constructed with
              a key-value store structure that use the ``frozenset`` type.

            - Fix issue with on-the-fly linear hash index build which was
              previously not correctly setting a set of integers.

Descriptor Generator Plugins

    - Fix issue with ``CaffeDescriptorGenerator`` where the GPU would not be
      appropriately used on a separate thread/process after initialization occurs on
      the main (or some other) thread.

Docker

    - IQR Playground

        - Updated README for better instruction on creating the docker image
          first.

    - Caffe image

        - Resolved an issue with upgrading pip for a newer version of matplotlib.

Documentation

    - Removed module mocking in sphinx ``conf.py`` as it has been shown to be
      brittle to changes in the source code.  If we isolate and document a
      use-case where the mocking becomes relevant again we can bring it back.

Misc.

    - Update requests and flask package version in ``requirements.txt`` and
      ``devops/docker/smqtk_wrapper_python/requirements.txt`` files due to
      GitHub security alert.

    - Updated package versions for packages in the ``requirements.docs.txt``
      requirements file.

Utilities

    - Fixed broken large file writing in Python 2, creating parity
      with Python 3.

    - Fixed ``iqr_app_model_generation.py`` script for the current state of
      SMQTK functionality.

    - Fixed double logging issue in ``python/smqtk/bin/classifyFiles.py``
      tool.

Web

    - IQR Search Demo App

        - Fixed input element autocomplete property value being set
          from disabled" to the correct value of "off".

        - Fix CSRF vulnerability in demo web application front-end.

        - Fixed sample configuration files for the current state of
          associated tools.
