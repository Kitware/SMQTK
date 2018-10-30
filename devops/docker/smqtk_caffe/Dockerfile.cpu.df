FROM ubuntu:16.04

# System Package dependencies
ENV TERM=xterm
RUN rm /bin/sh \
 && ln -s /bin/bash /bin/sh \
 && apt-get -y update \
 && apt-get -y install \
        # Basic dependencies
        build-essential python python-pip git cmake \
        # Caffe Dependencies
        libatlas-base-dev libatlas-dev libboost-all-dev libprotobuf-dev \
        protobuf-compiler libgoogle-glog-dev libgflags-dev libhdf5-dev \
 # Clean products of ``update``
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # Python dependencies for pycaffe
RUN pip install --upgrade pip
RUN pip install numpy scikit-image protobuf

#
# Caffe Install
#
ENV CAFFE_URL="https://github.com/BVLC/caffe.git" \
    CAFFE_VERSION="1.0" \
    PATH="/caffe/install/bin:${PATH}" \
    PYTHONPATH="/caffe/install/python:${PYTHONPATH}" \
    LD_LIBRARY_PATH="/caffe/install/lib:${LD_LIBRARY_PATH}"
RUN \
    # Python part installation location
    LOCAL_DIST_PACKAGES="$(python -c \
        "import distutils.sysconfig; \
         print(distutils.sysconfig.get_python_lib(prefix='/usr/local'))")" \
 && git clone https://github.com/BVLC/caffe.git /caffe/source \
 && cd /caffe/source \
 && git checkout ${CAFFE_VERSION} \
 # Create source/build directories for Caffe
 && mkdir -p /caffe/build \
 && cd /caffe/build \
 && cmake \
    # Need to specifically point to libatlas installed stuff.
    -DAtlas_BLAS_LIBRARY:PATH=/usr/lib/atlas-base/libatlas.so \
    -DAtlas_CBLAS_LIBRARY:PATH=/usr/lib/atlas-base/libcblas.so \
    -DAtlas_LAPACK_LIBRARY:PATH=/usr/lib/atlas-base/liblapack_atlas.so \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    # Turn off uneeded parts.
    -DUSE_LEVELDB:BOOL=OFF \
    -DUSE_LMDB:BOOL=OFF \
    -DUSE_OPENCV:BOOL=OFF \
    # Install to the system local space
    -DCMAKE_INSTALL_PREFIX:PATH=/usr/local \
    # CPU-Only build
    -DCPU_ONLY:BOOL=ON \
    /caffe/source \
 && make install -j $(nproc) \
 && mv "/usr/local/python/caffe" "${LOCAL_DIST_PACKAGES}/caffe" \
 # Clean up intermediate files.
 && rm -rf /caffe \
           /usr/local/python
