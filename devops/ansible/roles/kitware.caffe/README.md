kitware.caffe
=============

Ansible role to install [Caffe](https://github.com/BVLC/caffe).

Role Variables
--------------

The following variables may be overridden:

* `caffe_path`
  * Path to edownload and build Caffe in.
* `caffe_repo`
  * URL to the story to clone sm.
* `caffe_version`
  * Git commit/branch/tag for fetching Caffe.
* `caffe_install_prefix`
  * Path to the root to install Caffe into.
  * This must be somewhere root permission is not required.
* `caffe_virtual_env`
  * Install python bindings into this virtual env root.
  * This must be somewhere root permission is not required.

TODO: Data/model download options? Leave that to upload in girder?
