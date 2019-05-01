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

Web
* Added endpoints IQR headless service for expanded getter methods added to
  IqrSession class.

Fixes
-----

Algorithms
* DescriptorGenerator
  * Caffe
    * Fix configuration overrides to correctly handle configuration from JSON.
