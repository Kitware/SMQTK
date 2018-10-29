FROM kitware/smqtk/caffe:1.0-cpu

#
# Download Resnet-50 Caffe models.
# This reflects the same models used as the IQR Playground container.
#
RUN apt-get update -y \
 && apt-get install -y curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 # - Get MSRA ResNet-50 model files
 && mkdir -p /caffe/models/msra_resnet \
 && curl https://data.kitware.com/api/v1/item/5939a7828d777f16d01e4e5d/download \
        -o /caffe/models/msra_resnet/LICENSE.txt \
 && curl https://data.kitware.com/api/v1/item/5939a61e8d777f16d01e4e52/download \
        -o /caffe/models/msra_resnet/ResNet_mean.binaryproto \
 && curl https://data.kitware.com/api/v1/item/5939a6188d777f16d01e4e40/download \
        -o /caffe/models/msra_resnet/ResNet-50-deploy.prototxt \
 && curl https://data.kitware.com/api/v1/item/5939a6198d777f16d01e4e43/download \
        -o /caffe/models/msra_resnet/ResNet-50-model.caffemodel

#
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

COPY devops/docker/smqtk_classifier_service/default_server.cpu.json /configuration/server.json
COPY devops/docker/smqtk_classifier_service/entrypoint.sh /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
EXPOSE 5002
