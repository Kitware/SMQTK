VCD Feature Store
=================

The intent of the VCDStore structure is to provide an API to store
and retrieve feature vector information in a way that shields the user from
the underlying mechanic of *how* it is stored. Construction of the store
object, only implemented in python at the moment, uses a default implementation
and allows constructions with no explicit parameters. When there are multiple
back-ends available for use, a different backend may be specified at
construction, along with additional parameters specific to that back-end if
desired (for example, 2 additional parameters for the SQLite backend, described
below).

Currently, the default back-end unilizes SQLite3. Because this back-end is
completely python specific, it is not intended to be final, but a place-holder
for the time being. In the future, we would want to have the VCDStore be
accessible from other languages like MATLAB or C++. This back-end allows for
two additional parameters at VCDStore construction if desired: the path
to the SQLite database file or the path to the directory where one should be
(providing the actual path overrides the use of the directory path). For more
detail, view the actual implementation file
(SMQTK_Backend.VCDStore.implementations.SQLiteVCDStoreBackend).
