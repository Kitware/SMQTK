SMQTK External IQR Plugin
#########################
This plugin adds a small widget to folders in girder if certain metadata fields
are present for that folder's model.  If they exist, then a bar with a link
is generated at the top of the view that initializes the IQR configuration
with the remote server and opens a new tab/window to that service.

We expect the fields:

    smqtk_iqr_root
    smqtk_iqr

``smqtk_iqr_root`` should specify the URL to the root of a running
``IqrSearchDispatcher`` instance (see :command:`runApplication`).

``smqtk_iqr`` should be a valid configuration for a tab in
``IqrSearchDispatcher`` (see the ``__default__`` section in a generated config
for an example template).
