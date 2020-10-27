SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

Algorithms

* Added new algorithms ``RankRelevancy`` and ``RankRelevancyWithFeedback``.
  ``RankRelevancy`` is an overhaul of the existing ``RelevancyIndex`` algorithm
  and will eventually replace it.  ``RankRelevancyWithFeedback`` is a closely
  related algorithm that additionally provides feedback requests.

Documentation

* Update plugin related sphinx documentation content and examples.

Utils

* plugin

  * Revise the underpinnings of the utilities and
    :py:class:`smqtk.utils.plugin.Pluggable` mixin class to be more modular
    and involve less special rules.


Fixes
-----

Misc.

* Fix issues revealed with updating to use of mypy version 0.790.
