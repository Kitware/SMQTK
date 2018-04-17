SMQTK VIGILANT Pending Release Notes
====================================


Updates / New Features
----------------------

Fixes
-----

Tests

* Moved ``--cov`` options from pytest.ini file into the runner script.  This
  fixes debugger breakpointing in some IDEs.

Utils

* Fix ``ZeroDivisionError`` in ``smqtk.utils.bin_utils.report_progress``. Also
  deprecate this function.
