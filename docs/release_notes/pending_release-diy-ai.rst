SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

Algorithms

* Added new algorithms ``RankRelevancy`` and ``RankRelevancyWithFeedback``.
  ``RankRelevancy`` is an overhaul of the existing ``RelevancyIndex`` algorithm
  and will eventually replace it.  ``RankRelevancyWithFeedback`` is a closely
  related algorithm that additionally provides feedback requests.

  * An implementation ``RankRelevancyWithSupervisedClassifier`` is provided for
    ``RankRelevancy``, porting the existing
    ``SupervisedClassifierRelevancyIndex``.

  * An implementation ``RankRelevancyWithMarginSampledFeedback`` is provided
    for ``RankRelevancyWithFeedback``, supporting wrapping a ``RankRelevancy``
    instance for margin sampling.

CI

* Added test checking that a pending release notes files is updated on a merge
  request, otherwise it fails. The intent of this test is to remind
  contributors that they ought to be adding change notes.

Documentation

* Update plugin related sphinx documentation content and examples.

IQR

* Change over the previous use of the "RelevancyIndex" to the new
  "RankRelevancyWithFeedback" algorithm interface in the IqrSession class.
  Updates also reflected in the IQR web service and respective unit tests.

Utils

* plugin

  * Revise the underpinnings of the utilities and
    :py:class:`smqtk.utils.plugin.Pluggable` mixin class to be more modular
    and involve less special rules.


Fixes
-----

Misc.

* Fix issues revealed with updating to use of mypy version 0.790.
