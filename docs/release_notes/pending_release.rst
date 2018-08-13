SMQTK Pending Release Notes
===========================


Updates / New Features
----------------------

Algorithms

  - Classifier

    - Added `ClassifierCollection` support class. This assists with aggregating
      multiple SMQTK classifier implementations and applying one or more of
      those classifiers to input descriptors.

    - Split contents of the `__init__.py` file into multiple component files.
      This file was growing too large with the multiple abstract classes and a
      new utility class.

    - Changed `classify` abstract method to raise a `ValueError` instead of a
      `RuntimeError` upon being given an empty `DescriptorElement`.

    - Updated SupervisedClassifier abstract interface to use the template pattern
      with the train method. Now, implementing classes need to define
      ``_train``. The ``train`` method is not abstract anymore and calls the
      ``_train`` method after the input data consolidation.

    - Update API of classifier to support use of generic extra training parameters.

    - Updated libSVM classifier algorithm to weight classes based on the geometric
      mean of class counts divided by specific class count to more properly handle
      weighting even if there is class imbalance.

  - Hash Index

    - Made to be its own interface descending from `SmqtkAlgorithm` instead of
      `NearestNeighborsIndex`. While the functionality of a NN-Index and a
      HashIndex are very similar, all method interfaces are different in terms
      of the types they accept and return and the HashIndex implementation
      redefined and documented them to the point where there was no shared
      functionality.

    - Switched to using the template method for abstract methods.

    - Add update and remove methods to abstract interface. Implemented new
      interface methods in all subclasses.

    - Added model concurrency protection to implementations.

  - Nearest-Neighbors

    - Switched to using the template method for abstract methods.

    - Add update and remove methods to abstract interface. Implemented new
      interface methods in all subclasses.

    - Fix imports in FAISS wrapper module.

    - Added model concurrency protection to implementations.

    - FAISS

      - Add model persistence via optionally provided `DataElement`.

      - Fixed use of strings for python 2/3 compatibility.

      - Changed default factory string to "IVF1,Flat".

      - Added initial GPU support to wrapper. Currently only supports one GPU
        with explicit GPU ID specification.

Representations

  - Descriptor Index

    - Added `__contains__` method to abstract class to call the `has` method.
      This should usually be more efficient than scanning the iteration of the
      index which is what was happening before. For some implementations, at
      worst, the runtime for checking for inclusion will be the same (some
      implementations may *have* to iterate).

  - Descriptor Element

    - Interface

      - Hash value for an element is now only composed of UID value. This is an
        initial step in deprecating the use of the type-string property on
        descriptor elements.

      - Equality check between elements now just vector equality.

      - Added base implementation of `__getstate__` and `__setstate__`. Updated
        implementations to handle this as well as be backward compatible with
        their previous serialization formats.

      - Added a return of self to vector setting method for easier in-line
        setting after construction.

    - PostgreSQL

      - Updated to use PsqlConnectionHelper class.

  - KeyValueStore

    - Added `remove` and `remove_many` abstract methods to the interface. Added
      implementations to current subclasses.

    - Added `__getitem__` implementation.

Docker

  - Caffe

    - Updated docker images for CPU or GPU execution.

    - Updated Caffe version built to 1.0.0.

  - Added Classifier service docker images for CPU or GPU execution.

    - Inherits from the Caffe docker images.

    - Uses MSRA's ResNet-50 deep learning models.

  - IQR Playground

    - Updated configuration files.

    - Now only runs IQR RESTful service and IQR GUI web app (removed nearest-
      neighbors service).

    - Simplified source image mount point to `/images`.

    - Updated `run_container.*.sh` helper scripts.

    - Change deep-learning model used from AlexNet to MSRA's RestNet-50 model.

  - Versioning changes to, by default, encode date built instead of arbitrary
    separate versioning compared to SMQTK's versioning.

  - Classifier and IQR docker images now use the local SMQTK checkout on the host
    system instead of cloning from the internet.

IQR module

  - Added serialization load/save methods to the `IqrSession` class.

Scripts

  - `generate_image_transform`

    - Added stride parameter to image tile cropping feature to allow for more
      than just discrete, abutting tile cropping.

  - `runApplication`

    - Add ability to get more than individual app description from providing
      the `-l` option. Now includes the title portion of each web app's
      doc-string.

  - Added `smqtk-make-train-test-sets`

    - Create train/test splits from the output of the
      `compute_many_descriptors` tool, usually for training and testing a
      classifier.

Testing

  - Remove use of `nose-exclude` since there are now actual tests in the web
    sub-module.

  - Switch to using `pytest` as the test running instead of `nose`. Nose is
    now in "maintenance mode" and recommends a move to a different testing
    framework. Pytest is a popular a new powerful testing framework
    alternative with a healthy ecosystem of extensions.

  - Travis CI

    - Removed use of Miniconda installation since it wasn't being utilized in
      special way.

  - Added more tests for Flask-based web services.

Utilities module

  - Added mimetype utilities sub-module.

  - Added a web utilities module.

    - Added common function for making response Flask JSON instances.

  - Added an `iter_validation` utility submodule.

  - Plugin utilities

    - Updated plugin discovery function to be more descriptive as to why a
      module or class was ignored. This helps debugging and understanding why
      an implementation for an interface is not available at runtime.

  - PostgreSQL

    - Added locking to table creation upsert call.

  - Added probability utils submodule and initial probability adjustment function.

Web

  - Added new classifier service for managing multiple SMQTK classifier
    instances via a RESTful interface as well as describe arbitrary new data
    with the stored classifiers. This service also has the ability to take in
    saved IQR session states and train a new binary classifier from it.

    - Able to query the service with arbitrary data to be described and
      classified by one or more managed classifiers.

    - Able to get and set serializations of classifier models for archival.

    - Added example directory of show how to run and to interact with the
      classifier service via `curl`.

    - Optionally take a new parameter on the classify endpoint to adjust the
      precision/recall balance of results.

  - IQR Search Dispatcher (GUI web app)

    - Refactored to use RESTful IQR service.

    - Added GUI and JS to load an IQR state from file.

    - Update sample JSON configuration file at
      `python/smqtk/web/search_app/sample_configs/config.IqrSearchApp.json`.

    - Added `/is_ready` endpoint for determining that the service is alive.

  - IQR service

    - Added ability to an IQR state serialization into a session.

    - Added sample JSON configuration file to
      `python/smqtk/web/search_app/sample_configs/config.IqrRestService.json`.

    - Added `/is_ready` endpoint for determining that the service is alive.

    - Move class out of the `__init__.py` file and into its own dedicated file.

    - Make IQR state getter endpoint return a JSON containing the base64 of the
      state instead of directly returning the serialization bytes.

    - Added endpoints to update, remove from and query against the global
      nearest-neighbors index.

Fixes
-----

Algorithms

  - Nearest-Neighbor Index

    - LSH

      - Fix bug where it was reporting the size of the nested descriptor index
        as the size of the neighbor index when the actual index state is
        defined by the hash-to-uids key-value mapping.

Representations

  - DataElement

    - Fixed bug where `write_temp()` would fail if the `content_type()`
      was unknown (i.e. when it returned `None`).

  - Descriptor Index

    - PostgreSQL

      - Fix bug where an instance would create a table even though the
        `create_table` parameter was set to false.

  - Descriptor Elements

    - PostgreSQL implementation

      - Fix set_vector method to be able to take in sequences that are not
        explicitly numpy arrays.

  - KeyValue

    - PostgreSQL

      - Fix bug where an instance would create a table even though the
        `create_table` parameter was set to false.

Scripts

  - `classifier_model_validation`

    - Fixed confidence interval plotting.

    - Fixed confusion matrix plot value range to the [0,1] range which causes
      the matrix colors to have meaning across plots.

Setup.py

  - Add `smqtk-` to some scripts with camel-case names in order to cause them
    to be successfully removed upon uninstallation of the SMQTK package.

Tests

- Fixed ambiguous ordering check in libsvm-hik implementation of
  RelevancyIndex algorithm.

Web

  - IQR Search Dispatcher (GUI web app)

    - Fix use of `StringIO` to using `BytesIO`.

    - Protect against potential deadlock issues by wrapping intermediate code
      with try/finally clauses.

    - Fixed off-by-one bug in javascript `DataView` construction.

  - IQR Service

    - Gracefully handle no-positive-descriptors error on working index
      initialization.

    - Fix use of `StringIO` to using `BytesIO`.
