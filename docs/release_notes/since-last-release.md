Changes Since Last Release
==========================


Updates / New Features since v0.2.2
-----------------------------------

Tools / Scripts

  * Added optional global default config generation to ``summarizePlugins.py``

Web / Services

  * Added query sub-slice return option to NearestNeighborServiceServer web-app.


Fixes since v0.2.2
------------------

DescriptorElement

  * Fixed mutibility of stored descriptors in DescriptorMemoryElement
    implementation.

Tools / Scripts

  * Added ``Classifier`` interface plugin summarization to
    ``summarizePlugins.py``.

Web / Services

  * Fixed issue with IQR alerts not showing whitespace correctly.

  * Fixed issue with IQR reset not resetting everything, which caused the
    application to become unusable.
