SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

Algorithms
* DescriptorGenerator
  * Overhauled interface API to have the abstract method be a many-to-many
    iterator instead of the previous one-to-one signature.
  * Updated colordescriptor implementation for interface API update.
  * Updated caffe implementation for interface API update.
  * Updated KWCNN implementation for interface API update.

Compute Functions
* Updated ``smqtk.compute_functions.compute_many_descriptors`` to utilize new
  DescriptorGenerator API.

Docker
* Started use of docker-compose YAML file to organize image building.
* Added FAISS TPL image to be copied from by utilizing images.
* IQR "Playground"
  * Fixed compute test scripts to use updated DescriptorGenerator API.

Documentation
* Updated ``docs/algorithminterfaces.rst`` to reflect the new
  DescriptorGenerator API.
* Updated ``docs/algorithmmodels.rst`` to reflect the new DescriptorGenerator
  API.
* Updated the ``docs/examples/caffe_build_index.rst`` example to use the new
  DescriptorGenerator API.
* Updated the ``docs/examples/simple_feature_computation.rst`` example to use
  the new DescriptorGenerator API.

IQR
* Remove forcing of relevancy scores in ``refine`` when a result element is
  contained in the positive or negative exemplar or adjudication sets. This is
  because a user of an ``IqrSession`` instance can determine this intersection
  optionally outside of the class, so this forcing of the values is a loss of
  information.
* Added accessor functions to specific segments of the relevancy result
  predictions: positively adjudicated, negatively adjudicated and
  not-adjudicated elements.

Misc.
* Cleaned up various test warnings.

Representation
* AxisAlignedBoundingBox
  * Added ``intersection`` method.
* Data Element
  * Added PostgreSQL implementation.
* DataSet
  * Added PostgreSQL implementation, storing data representation natively in
    the database.
* DetectionElement
  * Added individual component accessors.
* Renamed "DescriptorIndex" to "DescriptorSet" in order to better represent
  what the structure and API represents. "Index" can carry the connotation that
  more is happening within the structure than actually is.

Tests
* Updated colordescriptor DescriptorGenerator tests to "skip" when deemed not
  available so that the tests are not just hidden when the optional
  dependencies are not present.
* Updated dummy classes used in classifier service unit tests to match the new
  DescriptorGenerator API.
* Update IQR service unit tests stub class for the new DescriptorGenerator API
  and iteration properties.
* Updated various class unit tests to make use of new configuration test helper
  function.

Tools / Scripts
* Updated the ``smqtk.bin.classifyFiles`` tool to use the new
  DescriptorGenerator API.
* Updated the ``smqtk.bin.computeDescriptor`` tool to use the new
  DescriptorGenerator API.
* Updated the ``smqtk.bin.iqr_app_model_generation`` tool to use the new
  DescriptorGenerator API.
* Updated some old MEMEX scripts to use the new DescriptorGenerator API.

Utils
* Added additional description capability to ProgressReporter.
* Added a return of self in the ``ContentTypeValidator.raise_valid_element()``
  method.
* Added helper function for testing Configurable mixing instance functionality.
* Promoted service proxy helper class from IQR service server to a general web
  utility.
* Update random character generator to use ``random.SystemRandom`` which, at
  least for Posix systems, uses a source suitable for cryptographic purposes.

Web
* Added endpoints IQR headless service for expanded getter methods added to
  IqrSession class.
* Changed IQR web service endpoint to retrieve nearest-neighbors to a GET
  method instead of the previous POST method, as the previous method did not
  make sense for the request being made.
* Fixed usage of DescriptorGenerator instances in the classifier service for
  the API update.
* Updated ``smqtk.web.descriptor_service`` to use the new DescriptorGenerator
  API.
* Updated ``smqtk.web.iqr_service`` to use the new DescriptorGenerator API.
* Updated ``smqtk.web.nearestneighbor_service`` to use the new
  DescriptorGenerator API.

Fixes
-----

Algorithms
* DescriptorGenerator
  * Caffe
    * Fix configuration overrides to correctly handle configuration from JSON.
    * Coerce unicode arguments to Net constructor to strings (or bytes in
      python 3).
    * Fixed numpy load call to explicitly allow loading pickled components due
      to a parameter default change in numpy version 1.16.3.
* HashIndex
  * SkLearnBallTreeHashIndex
    * Fixed numpy load call to explicitly allow loading pickled components due
      to a parameter default change in numpy version 1.16.3.
* ImageMatrixObjectDetector
  * Add ``abstractmethod`` decorator to intermediate implementation of
    ``get_config`` method.

Tests
* Fixed PostgreSQL KeyValueStore implementation unit test that became
  non-deterministic in Python 3+.

Utilities
* Fixed issue with ProgressReporter when reporting before the first interval
  period.
* Fixed issue with ``smqtk.utils.parallel.parallel_map`` function where it
  could hang during threading-mode when a keyboard interrupt occurred.
* Fixed incorrectly calling the module-level debug logging function to use the
  locally passed logger, cleaning up a duplicate logging issue.

Web
* Classifier Service
  * Fix configuration of CaffeDescriptorGenerator.
* IQR Service
  * Fix configuration of CaffeDescriptorGenerator.
