SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

Docker
* Started use of docker-compose YAML file to organize image building.
* Added FAISS TPL image to be copied from by utilizing images.

IQR
* Remove forcing of relevancy scores in ``refine`` when a result element is
  contained in the positive or negative exemplar or adjudication sets. This is
  because a user of an ``IqrSession`` instance can determine this intersection
  optionally outside of the class, so this forcing of the values is a loss of
  information.
* Added accessor functions to specific segments of the relevancy result
  predictions: positively adjudicated, negatively adjudicated and
  not-adjudicated elements.

Representation
* DetectionElement
  * Added individual component accessors.

Utils
* Added additional description capability to ProgressReporter.

Web
* Added endpoints IQR headless service for expanded getter methods added to
  IqrSession class.

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

Web
* Classifier Service
  * Fix configuration of CaffeDescriptorGenerator.
* IQR Service
  * Fix configuration of CaffeDescriptorGenerator.
