ARG CUDA_RUNTIME_IMAGE_TAG
ARG CAFFE_COPY_FROM_IMAGE
##############################################################################
# Base system packages and depedencies
#
FROM ${CAFFE_COPY_FROM_IMAGE} AS caffe_image
FROM nvidia/cuda:${CUDA_RUNTIME_IMAGE_TAG} AS smqtk-cpu

SHELL ["/bin/bash", "-c"]

# System Package dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
 && apt-get -y install \
        cmake=3.10.2-1ubuntu2.18.04.1 \
        curl=7.58.0-2ubuntu3.10 \
        less=487-0.1 \
        parallel=20161222-1 \
        python3-dev=3.6.7-1~18.04 \
        python3-pip=9.0.1-2.3~ubuntu1.18.04.3 \
        vim=2:8.0.1453-1ubuntu1.3 \
        # Caffe TPL deps
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
        # Caffe python deps
        numpy==1.18.4 scikit-image==0.16.2 protobuf==3.11.3

###############################################################################
# COPY in Caffe wheel and install
#
# Model downloads
# - Doing this before because we know that the caffe image reference is
#   variable; don't want to have to download twice.
RUN mkdir -p /caffe/models/msra_resnet \
 && cd /caffe/models/msra_resnet \
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
# SMQTK install
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
 && make install -j12 \
 && cd / \
 && rm -rf smqtk \
 # Create directory for standard configuration file mounting
 && mkdir /configuration

ADD devops/docker/smqtk_classifier_service/default_server.cpu.json \
    /configuration/server.json
ADD devops/docker/smqtk_classifier_service/entrypoint.sh \
    /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
EXPOSE 5002

###############################################################################
# Add GPU variant specific files and configuration
#
FROM smqtk-cpu AS smqtk-gpu
ADD devops/docker/smqtk_classifier_service/default_server.gpu.json \
    /configuration/server.json
