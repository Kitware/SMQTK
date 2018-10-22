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

Utilities
    - Fixed broken large file writing in Python 2, creating parity
      with Python 3.

Web
    - IQR Search Demo App
        - Fixed input element autocomplete property value being set
          from disabled" to the correct value of "off".
