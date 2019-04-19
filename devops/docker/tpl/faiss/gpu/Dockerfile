FROM nvidia/cuda:8.0-cudnn7-devel-centos7

RUN yum install -y epel-release \
 && yum install -y \
      git-svn-1.8.3.1-20.el7.x86_64 \
      make-3.82-23.el7.x86_64 \
      openblas-devel-0.3.3-2.el7.x86_64 \
      python-devel-2.7.5-77.el7_6.x86_64 \
      python2-pip-8.1.2-8.el7.noarch \
      swig3-3.0.12-17.el7.x86_64
RUN pip install numpy==1.16.2

WORKDIR /faiss/source
# TODO: bring out to staging?
RUN git clone https://github.com/facebookresearch/faiss.git /faiss/source \
 && git checkout v1.5.1

# Apply patch for v1.5.1
COPY patch/v1.5.1 /faiss/source

#
# Configure and Build FAISS library and bindings.
#
ARG CUDA_ARCH
ENV PYTHONPATH="/faiss/install/lib/python2.7/site-packages"
RUN ./configure \
      --with-cuda=/usr/local/cuda \
      --with-cuda-arch="${CUDA_ARCH}" \
      --prefix=/faiss/install \
 && make install -j$(nproc) \
    # mkdir required apparently because of setting PYTHONPATH
 && mkdir -p ${PYTHONPATH} \
 && make -C python install -j$(nproc)
