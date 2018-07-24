FROM kitware/smqtk/caffe:1.0-gpu-cuda8.0-cudnn6
MAINTAINER paul.tunison@kitware.com

# System Package dependencies
RUN apt-get -y update \
 && apt-get -y install lsb-release git cmake curl vim less parallel sudo \
 # PostgreSQL 9.4 Installation
 # - Adding commonly used pg_ctl and postgres command links to bin for more
 #   convenient access.
 && echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -sc)-pgdg main" \
    >/etc/apt/sources.list.d/pgdg.list \
 && curl --silent https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    | apt-key add - \
 && apt-get -y update \
 && apt-get -y install postgresql-9.4 postgresql-server-dev-9.4 \
 && ln -s /usr/share/postgresql-common/pg_wrapper /usr/bin/pg_ctl \
 && ln -s /usr/share/postgresql-common/pg_wrapper /usr/bin/postgres \
 # MongoDB Installation
 && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927 \
 && echo "deb http://repo.mongodb.org/apt/ubuntu $(lsb_release -sc)/mongodb-org/3.2 multiverse" \
    >/etc/apt/sources.list.d/mongodb-org-3.2.list \
 && apt-get -y update \
 && apt-get -y install mongodb-org \
 # Some additional system+python packages for convenience and SMQTK optional
 # deps.
 && pip install ipython psycopg2 file-magic \
 # Clean up apt resources.
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


# SMQTK Installation
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
 && rm -rf smqtk


# Add ``smqtk`` user and add sudo privilege
RUN useradd -mr -s /bin/bash smqtk \
 # sudo permission for modifying permissions at runtime (see entrypoint.sh).
 && echo "smqtk ALL=(ALL:ALL) NOPASSWD:ALL" >>/etc/sudoers \
 && mkdir /images \
 && chown smqtk:smqtk /images
USER smqtk
WORKDIR /home/smqtk


# User Space ##################################################################

# Get MSRA ResNet-50 model files
RUN mkdir -p caffe/msra_resnet \
 && curl https://data.kitware.com/api/v1/item/5939a7828d777f16d01e4e5d/download \
        -o caffe/msra_resnet/LICENSE.txt \
 && curl https://data.kitware.com/api/v1/item/5939a61e8d777f16d01e4e52/download \
        -o caffe/msra_resnet/ResNet_mean.binaryproto \
 && curl https://data.kitware.com/api/v1/item/5939a6188d777f16d01e4e40/download \
        -o caffe/msra_resnet/ResNet-50-deploy.prototxt \
 && curl https://data.kitware.com/api/v1/item/5939a6198d777f16d01e4e43/download \
        -o caffe/msra_resnet/ResNet-50-model.caffemodel

# Expected user-space directories.
RUN mkdir -p data/{models,configs,logs,db.psql,db.mongo} \
 && ln -s /images data/images


COPY devops/docker/smqtk_iqr_playground/entrypoint.sh \
     devops/docker/smqtk_iqr_playground/descr_comp_test.cpu.py \
     devops/docker/smqtk_iqr_playground/descr_comp_test.gpu.py \
     /home/smqtk/
COPY devops/docker/smqtk_iqr_playground/default_confs/* \
     devops/docker/smqtk_iqr_playground/default_confs/gpu/* \
     /home/smqtk/data/configs/
ENTRYPOINT ["/home/smqtk/entrypoint.sh"]
EXPOSE 5000 5001
