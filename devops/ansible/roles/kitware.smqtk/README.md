kitware.smqtk
=============

Ansible role to install [SMQTK](https://github.com/Kitware/SMQTK).


Role Variables
--------------

The following variables may be overridden:

* `smqtk_path`: Path to download and build SMQTK in.
* `smqtk_repo`: URL to the repository to clone from.
* `smqtk_version`: Git commit/branch/tag for fetching smqtk.
* `smqtk_virtualenv`: Path to the python virtual environment to install SMQTK into.
* `smqtk_iqr_secret_key`: String to use as server Flask secret key.
                          This should be changed from the default.
* `smqtk_iqr_host`: Host address to listen to.
* `smqtk_iqr_port`: Port to use


Dependencies
------------

This role depends on the following roles:

* `Stouts.monogodb`
* `ANXS.postgresql`
* `kitware.caffe`
