Docker build of Facebook AI Research's FAISS library and python bindings.
This image is intended to not be included directly but for built components to
be copied out of this image.

System dependencies used:
  - FROM nvidia/cuda:8.0-cudnn7-devel-centos7
  - YUM installed:
      git-svn (1.8.3.1-20)
      make (3.82-23)
      openblas-devel (0.3.3-2)
      python-devel (2.7.5-77)
      python2-pip (8.1.2-8)
      swig3 (3.0.12-17)

FAISS install root inside built image is ``/faiss/install``.

# Patches
Patches to the FAISS source tree for specific versions of faiss are located in
the ``patch`` directory separated by version tag. The directory contents for a
specific version should be directly copied into the FAISS source tree in the
Docker image during build.
