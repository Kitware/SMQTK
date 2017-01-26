FROM nvidia/caffe

RUN apt-get update
RUN apt-get install -y git python-pip cmake curl wget \
                       libatlas-base-dev libatlas-dev \
                       libboost1.55-all-dev \
                       libprotobuf-dev protobuf-compiler \
                       libgoogle-glog-dev libgflags-dev \
                       libhdf5-dev \
                       libopencv-dev \
                       liblmdb-dev \
                       libleveldb-dev \
                       libsnappy-dev \
                       gfortran libgtest-dev libpq-dev
RUN pip install psycopg2 file-magic


RUN git clone https://github.com/Kitware/SMQTK.git /SMQTK

WORKDIR /SMQTK
RUN mkdir build

WORKDIR build
RUN cmake .. && make -j4 && make install

WORKDIR /SMQTK
RUN pip install -r requirements.txt
RUN pip install .

# Fetch the caffe source in order to use the provided models
# which aren't distributed with this container
RUN mkdir /caffe-src
RUN git clone https://github.com/NVIDIA/caffe.git /caffe-src
RUN /caffe-src/data/ilsvrc12/get_ilsvrc_aux.sh
