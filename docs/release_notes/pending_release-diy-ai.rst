SMQTK DIY-AI Pending Release Notes
==================================


Updates / New Features
----------------------

CI

* Deprecate the use of Drone, changing to the use of GitLab runner
  configuration for use in that context.

* Added flake8 and mypy checks to the gitlab task list for format and type
  checking, respectively.

Misc.

* Updated requirements file layout. Now there is a single ``requirements.txt``
  in the root directory, with other requirements files within the
  ``requirements/`` directory.

* Deprecate the use of the ``setup.cfg`` file.

Utils

* bits

  * Removed unused JIT-decorated functions, which also removed unused optional
    dependency on numba.


Fixes
-----

Misc.

* Fixed various small formatting issues raised by new use of flake8.

* Fixed various small issues raised by new use of mypy.
