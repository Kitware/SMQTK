FROM ubuntu:trusty
MAINTAINER paul.tunison@kitware.com

# System Package dependencies
ENV TERM=xterm
RUN rm /bin/sh \
 && ln -s /bin/bash /bin/sh \
 && apt-get -y update \
 && apt-get -y install git cmake curl vim less parallel


# System Space #################################################################


# PostgreSQL installation
# - Adding commonly used pg_ctl and postgres command links to bin
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ trusty-pgdg main" \
    >/etc/apt/sources.list.d/pgdg.list \
 && curl --silent https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
 && apt-get -y update && apt-get -y install postgresql-9.4 \
 && ln -s /usr/share/postgresql-common/pg_wrapper /usr/bin/pg_ctl \
 && ln -s /usr/share/postgresql-common/pg_wrapper /usr/bin/postgres


# MongoDB installation
# - Not using standard system install because that would cause a boost install
#   version conflict with caffe. Caffe is more imporant here.
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10 \
 && echo "deb http://repo.mongodb.org/apt/ubuntu $(lsb_release -sc)/mongodb-org/3.0 multiverse" \
    >/etc/apt/sources.list.d/mongodb-org-3.0.list \
 && apt-get -y update \
 && apt-get -y install mongodb-org


# Caffe system deps
RUN apt-get -y install \
    libatlas-base-dev libatlas-dev libboost1.55-all-dev libprotobuf-dev \
    protobuf-compiler libgoogle-glog-dev libgflags-dev libhdf5-dev
    # Optional: libopencv-dev liblmdb-dev libleveldb-dev libsnappy-dev
    # - See USE_* variabled in Caffe CMake configuration below.


# User Space ###################################################################

# Add ``smqtk`` user and add sudo privilege
RUN useradd -mr -s /bin/bash smqtk \
 && echo "smqtk ALL=(ALL:ALL) NOPASSWD:ALL" >>/etc/sudoers
USER smqtk
WORKDIR /home/smqtk


# Miniconda setup
ENV PATH="/home/smqtk/miniconda/bin:${PATH}"
RUN curl --insecure \
    https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
    >miniconda2-latest-linux-x86_64.sh \
 && bash miniconda2-latest-linux-x86_64.sh -b -p ${HOME}/miniconda \
 && rm miniconda2-latest-linux-x86_64.sh \
 && conda update --all
# - Some python niceties
RUN pip install ipython


# Caffe installation
# - git hash is master as of 2016/10/05
ENV CAFFE_URL="https://github.com/BVLC/caffe.git" \
    CAFFE_VERSION=4ba654f5c88c36ee8ba53964b7faf25c6d7010b4
# - Python deps
RUN pip install numpy scikit-image protobuf
# - Get the source code
RUN git clone https://github.com/BVLC/caffe.git caffe/source \
 && cd caffe/source \
 && git checkout ${CAFFE_VERSION}
# - Build, linking to deps
RUN mkdir caffe/build \
 && cd caffe/build \
 && cmake \
    -DAtlas_BLAS_LIBRARY:PATH=/usr/lib/libatlas.so \
    -DAtlas_CBLAS_LIBRARY:PATH=/usr/lib/libcblas.so \
    -DAtlas_LAPACK_LIBRARY:PATH=/usr/lib/liblapack_atlas.so \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCPU_ONLY:BOOL=ON \
    -DPYTHON_EXECUTABLE:PATH=${HOME}/miniconda/bin/python2.7 \
    -DPYTHON_INCLUDE_DIR:PATH=${HOME}/miniconda/include/python2.7 \
    -DPYTHON_INCLUDE_DIR2:PATH=${HOME}/miniconda/include/python2.7 \
    -DPYTHON_LIBRARY:PATH=${HOME}/miniconda/lib/libpython2.7.so \
    -DUSE_CUDNN:BOOL=OFF \
    -DUSE_LEVELDB:BOOL=OFF \
    -DUSE_LMDB:BOOL=OFF \
    -DUSE_OPENCV:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=${HOME}/caffe/install \
    ${HOME}/caffe/source \
 && make -j12
# - "Install" python component with .pth file
RUN echo "${HOME}/caffe/source/python" >miniconda/lib/python2.7/site-packages/caffe.pth
# - Get basic model files
RUN curl https://data.kitware.com/api/v1/item/57dae22f8d777f10f26a2a85/download \
        -o caffe/source/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
 && curl https://data.kitware.com/api/v1/item/57e2f3fd8d777f10f26e532b/download \
        -o caffe/source/models/bvlc_alexnet/deploy.prototxt \
 && curl https://data.kitware.com/api/v1/item/57dae0a88d777f10f26a2a81/download \
        -o caffe/source/data/ilsvrc12/imagenet_mean.binaryproto


# SMQTK local installation
# - python requirements retreived when pip installing smqtk instead of from
#   ``requirements.txt`` file.
ENV SMQTK_VERSION="gur/vigilant/docker-playground-iqr" \
    SMQTK_URL="https://kwgitlab.kitware.com/computer-vision/SMQTK.git"
RUN git clone -b ${SMQTK_VERSION} ${SMQTK_URL} smqtk/source
RUN mkdir smqtk/build \
 && cd smqtk/build \
 && cmake \
    -DCMAKE_INSTALL_PREFIX:PATH=${HOME}/miniconda \
    -DSMQTK_PIP_INSTALL_FLAGS:STRING="-e" \
    ${HOME}/smqtk/source \
 && make install -j12
# - added optional plugin requirements
RUN sudo apt-get -y install postgresql-server-dev-9.4 \
 && pip install psycopg2
# Expected volume mount directories
RUN mkdir -p data/{images,models,configs,logs,db.psql,db.mongo}

ADD entrypoint.sh /home/smqtk/
ADD default_confs/* /home/smqtk/data/configs/
ENTRYPOINT ["/home/smqtk/entrypoint.sh"]
EXPOSE 5000
