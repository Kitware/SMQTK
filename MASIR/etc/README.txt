Etc configuration files
=======================

users.json
----------
Listing of allowed users and their account information (username, password,
etc.)

models.json
-----------
Models available for use within the system (i.e. the things that we have
generated data models for). Paths are relative to the masir_config.DIR_DATA
directory where everything is (or should be) stored. (paths could become lists
if cross platform support is desired, would then be concatenated with
os.path.join).