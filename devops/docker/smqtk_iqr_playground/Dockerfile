ARG CUDA_RUNTIME_IMAGE_TAG
ARG CAFFE_COPY_FROM_IMAGE
ARG FAISS_COPY_FROM_IMAGE
##############################################################################
# Base system packages and depedencies
#
FROM ${FAISS_COPY_FROM_IMAGE} AS faiss_image
FROM ${CAFFE_COPY_FROM_IMAGE} AS caffe_image
FROM nvidia/cuda:${CUDA_RUNTIME_IMAGE_TAG} AS smqtk-cpu
MAINTAINER paul.tunison@kitware.com

SHELL ["/bin/bash", "-c"]

# System Package dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
 && apt-get -y install \
        cmake=3.10.2-1ubuntu2.18.04.1 \
        curl=7.58.0-2ubuntu3.8 \
        less=487-0.1 \
        parallel=20161222-1 \
        python3-dev=3.6.7-1~18.04 \
        python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
        sudo=1.8.21p2-3ubuntu1.2 \
        vim=2:8.0.1453-1ubuntu1.3 \
        # PostgreSQL 10.12 Installation
        postgresql-10=10.12-0ubuntu0.18.04.1 \
        # MongoDB Installation
        mongodb=1:3.6.3-0ubuntu1.1 \
        # Caffe/FAISS TPL deps
        libboost-python1.65.1=1.65.1+dfsg-0ubuntu5 \
        libboost-system1.65.1=1.65.1+dfsg-0ubuntu5 \
        libboost-thread1.65.1=1.65.1+dfsg-0ubuntu5 \
        libgoogle-glog0v5=0.3.5-1 \
        libgflags2.2=2.2.1-1 \
        libhdf5-100=1.10.0-patch1+docs-4 \
        libprotobuf10=3.0.0-9.1ubuntu1 \
        libopenblas-base=0.2.20+ds-4 \
 # Clean up apt resources.
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
# Python package requirements for Caffe/FAISS + deps for SMQTK parts for
# postgresql/mongo. Add some nice-to-haves in there, too.
RUN pip3 install \
        # Update version of six
        six==1.14.0 \
        # Caffe/FAISS python deps
        numpy==1.18.4 scikit-image==0.16.2 protobuf==3.11.3 \
        # PostgreSQL python client
        psycopg2-binary==2.8.5 \
        # Nice-to-haves
        file-magic==0.4.0 \
        ipython==7.14.0

###############################################################################
# COPY in FAISS wheel and install
#
ARG FAISS_VERSION
COPY --from=faiss_image \
     /opt/faiss/python/dist/faiss-${FAISS_VERSION}-py3-none-any.whl /
RUN pip3 install /faiss-${FAISS_VERSION}-py3-none-any.whl \
 && rm /faiss-${FAISS_VERSION}-py3-none-any.whl

###############################################################################
# COPY in Caffe wheel and install
#
# Model downloads
# - Doing this before because we know that the caffe image reference is
#   variable; don't want to have to download twice.
RUN mkdir -p /downloads/caffe/msra-resnet-50 \
 && cd /downloads/caffe/msra-resnet-50 \
 && curl https://data.kitware.com/api/v1/item/5939a7828d777f16d01e4e5d/download \
        -o LICENSE.txt \
 && curl https://data.kitware.com/api/v1/item/5939a61e8d777f16d01e4e52/download \
        -o ResNet_mean.binaryproto \
 && curl https://data.kitware.com/api/v1/item/5939a6188d777f16d01e4e40/download \
        -o ResNet-50-deploy.prototxt \
 && curl https://data.kitware.com/api/v1/item/5939a6198d777f16d01e4e43/download \
        -o ResNet-50-model.caffemodel

COPY --from=caffe_image /caffe-1.0-py3-none-any.whl /
RUN pip3 install /caffe-1.0-py3-none-any.whl \
 && rm /caffe-1.0-py3-none-any.whl

###############################################################################
# SMQTK Installation
#
COPY docs /smqtk/source/docs
COPY etc /smqtk/source/etc
COPY python /smqtk/source/python
COPY src /smqtk/source/src
COPY TPL /smqtk/source/TPL
COPY CMakeLists.txt LICENSE.txt pytest.* README.md setup.* setup_env.* VERSION \
     /smqtk/source/
RUN mkdir /smqtk/build \
 && cd /smqtk/build \
 && cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_INSTALL_PREFIX:PATH=/usr/local \
    -DSMQTK_BUILD_FLANN:BOOL=OFF \
    /smqtk/source \
 && make install -j$(nproc) \
 && cd / \
 && rm -rf smqtk

# Shim to expose legacy postgresql CLI tools
# TODO: Clean this up / modernize / don't be running a local database
RUN ln -s /usr/share/postgresql-common/pg_wrapper /usr/local/bin/pg_ctl \
 && ln -s /usr/share/postgresql-common/pg_wrapper /usr/local/bin/postgres

# Add ``smqtk`` user
RUN useradd -mr -s /bin/bash smqtk \
 # sudo permission for modifying permissions at runtime (see entrypoint.sh).
 # TODO: Remove this eventually since its defeating the point of having a user.
 && echo "smqtk ALL=(ALL:ALL) NOPASSWD:ALL" >>/etc/sudoers \
 && mkdir -m 0555 /images

USER smqtk
WORKDIR /home/smqtk

# Setup expected user-space directories.
RUN mkdir -p data/{models,configs,logs,db.psql,db.mongo} \
 && ln -s /images data/images \
    # Link to Caffe downloads
 && mkdir caffe \
 && ln -s /downloads/caffe/msra-resnet-50 caffe/msra_resnet

# Configuration for Matplotlib to just use the Agg backend, not Tkinter
COPY --chown=smqtk:smqtk \
     devops/docker/smqtk_iqr_playground/matplotlibrc \
     .config/matplotlib/

# Copy in general entrypoint and CPU-variant test script
COPY --chown=smqtk:smqtk \
     devops/docker/smqtk_iqr_playground/entrypoint.sh \
     devops/docker/smqtk_iqr_playground/descr_comp_test.cpu.py \
     /home/smqtk/

# Copy in general and CPU-variant configs
COPY --chown=smqtk:smqtk \
     devops/docker/smqtk_iqr_playground/default_confs/* \
     devops/docker/smqtk_iqr_playground/default_confs/cpu/* \
     /home/smqtk/data/configs/

ENTRYPOINT ["/home/smqtk/entrypoint.sh"]
EXPOSE 5000 5001

###############################################################################
# Add GPU variant specific files and configuration
#
FROM smqtk-cpu AS smqtk-gpu
# Add GPU-variant test script
COPY --chown=smqtk:smqtk \
     devops/docker/smqtk_iqr_playground/descr_comp_test.gpu.py \
     /home/smqtk/
# Overwrite CPU-variance configs with the GPU variants.
COPY --chown=smqtk:smqtk \
     devops/docker/smqtk_iqr_playground/default_confs/gpu/* \
     /home/smqtk/data/configs/
