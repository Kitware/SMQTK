Docker image to build the BAIR/BVLC Caffe 1.0 libraries and python package

This image is intended to be used with a "COPY --from=..." statement in docker
images that would like to use the built Caffe libraries or statically-built
python "module".

Runtime depedencies required:
    docker (GPU-only): Base image descending from the development or runtime
        `nvidia/cuda` image of the same CUDA Version this image was built
        against (see CUDA_DEVEL_IMAGE_TAG arg used).
    apt:
      libboost-python1.65.1=1.65.1+dfsg-0ubuntu5
      libboost-system1.65.1=1.65.1+dfsg-0ubuntu5
      libboost-thread1.65.1=1.65.1+dfsg-0ubuntu5
      libgoogle-glog0v5=0.3.5-1
      libgflags2.2=2.2.1-1
      libhdf5-100=1.10.0-patch1+docs-4
      libprotobuf10=3.0.0-9.1ubuntu1
      libopenblas-base=0.2.20+ds-4
      python3=3.6.7-1~18.04
      python3-pip=9.0.1-2.3~ubuntu1.18.04.1
    pip:
      numpy==1.18.4
      scikit-image==0.16.2
      protobuf==3.11.3

COPY locations:
    A local installation tree can be found in `/opt/caffe/install/`.
    This includes the `/opt/caffe/install/python/caffe` python package
    directory.
    A wheel is additionally generated and is located at
    `/caffe-1.0-py3-none-any.whl`.

NOTE: The Ubuntu package manager "caffe" installs are not being utilized
because they attempt to install a lot of dependencies. The local use-case of
this image is to use the python bindings that can be built.
