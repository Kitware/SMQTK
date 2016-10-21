FROM ubuntu:14.04

# System setup and package installation
RUN rm /bin/sh \
 && ln -s /bin/bash /bin/sh \
 && apt-get -y update \
 && apt-get -y install git cmake curl wget \
                       libatlas-base-dev libatlas-dev \
                       libboost1.55-all-dev \
                       libprotobuf-dev protobuf-compiler \
                       libgoogle-glog-dev libgflags-dev \
                       libhdf5-dev \
                       libopencv-dev \
                       liblmdb-dev \
                       libleveldb-dev \
                       libsnappy-dev

# Caffe installation
# - Download caffe master as of 2016/05/20
RUN mkdir /caffe /caffe/models /caffe/build \
 && curl -L https://github.com/BVLC/caffe/archive/e79bc8f.tar.gz >caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a.tar.gz \
 && tar -xzf caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a.tar.gz \
 && mv caffe-e79bc8f1f6df4db3a293ef057b7ca5299c01074a /caffe/source
# - Build, linking to deps
RUN cd /caffe/build \
 && cmake \
    -DAtlas_BLAS_LIBRARY:PATH=/usr/lib/libatlas.so \
    -DAtlas_CBLAS_LIBRARY:PATH=/usr/lib/libcblas.so \
    -DAtlas_LAPACK_LIBRARY:PATH=/usr/lib/liblapack_atlas.so \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCPU_ONLY:BOOL=ON \
    -DPYTHON_EXECUTABLE:PATH=/miniconda/bin/python2.7 \
    -DPYTHON_INCLUDE_DIR:PATH=/miniconda/include/python2.7 \
    -DPYTHON_INCLUDE_DIR2:PATH=/miniconda/include/python2.7 \
    -DPYTHON_LIBRARY:PATH=/miniconda/lib/libpython2.7.so \
    -DUSE_CUDNN:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=/caffe/install \
    /caffe/source \
 && make install -j12 \
 && cd \
 && rm -r /caffe/source /caffe/build
ENV PATH="/caffe/install/bin:${PATH}" \
    PYTHONPATH="/caffe/install/python:${PYTHONPATH}"
