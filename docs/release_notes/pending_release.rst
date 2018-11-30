SMQTK Pending Release Notes
===========================


Updates / New Features
----------------------


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

Misc.

    - Update requests and flask package version in ``requirements.txt`` and
      ``devops/docker/smqtk_wrapper_python/requirements.txt`` files due to
      GitHub security alert.

Utilities

    - Fixed broken large file writing in Python 2, creating parity
      with Python 3.

Web

    - IQR Search Demo App

        - Fixed input element autocomplete property value being set
          from disabled" to the correct value of "off".

        - Fix CSRF vulnerability in demo web application front-end.
