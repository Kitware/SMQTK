SMQTK MAD-E Pending Release Notes
=================================

Updates / New Features
----------------------

Utils

* Added to ``Pluggable`` interface the ``get_impls`` method, replacing the
  separate ``get_*_impls`` functions defined for each interface type.  Removed
  previous ``get_*_impls`` functions from algorithm and representation
  interfaces, adjusting tests and utilities as appropriate.
