SMQTK Pending Release Notes
===========================


Updates / New Features since v0.8.0
-----------------------------------


Fixes since v0.8.0
------------------

Descriptor Index Plugins

* Fix bug in PostgreSQL plugin where the helper class was not being called
  appropriately.

Utilities

* Fix bug in PostgreSQL connection helper where the connection object was
  being called upon when it may not have been initialized.
