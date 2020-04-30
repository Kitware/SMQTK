#
# Docker Image to build the FAISS library and python 3 bindings+wheel
#
# In most use-cases you will want to `COPY --from` an image built from this,
# and usually the python3 wheel
#
# Runtime depedencies required:
#   docker: Base image descending from the development or runtime `nvidia/cuda`
#       image of the same CUDA Version this image was built against (see
#       CUDA_DEVEL_IMAGE_TAG arg used).
#   apt:
#     libopenblas-base=0.2.20+ds-4
#     python3=3.6.7-1~18.04
#     python3-pip=9.0.1-2.3~ubuntu1.18.04.1
#   pip:
#     numpy==1.18.4
#
# When using `COPY --from` this image:
#   Built C libraries are located within the source/build root ``/opt/faiss``.
#
#   A python wheel is also built and may be coped from:
#     /opt/faiss/python/dist/faiss-{VERSION}-py3-none-any.whl
#
#   The bundled swig bindings library should only require the libraries from
#   runtime requirements listed above and the CUDA runtime libraries provided
#   by a CUDA-version matching `nvidia/cuda` docker image.
#
# When Extending from this image:
#   At least the next-to-last block that is currently commented out should be
#   uncommented. This will
#   FAISS libraries and python package are installed into ``/opt/faiss``.
#   Note that not a lot of cleanup has occurred in this image and this is based
#   on a development CUDA docker image, so sizes will probably be large.
#
ARG CUDA_DEVEL_IMAGE_TAG
FROM nvidia/cuda:$CUDA_DEVEL_IMAGE_TAG AS base

RUN apt-get -y update \
 && apt-get -y install \
      git=1:2.17.1-1ubuntu0.7 \
      libopenblas-dev=0.2.20+ds-4 \
      python3-dev=3.6.7-1~18.04 \
      python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
      swig=3.0.12-1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install numpy==1.18.3

WORKDIR /opt/faiss

# Acquire the source code, currently from the github repo.
# - If an *offline* build is required, replace this with a copy and unpack of
#   some source archive from the host.
ARG FAISS_VERSION
RUN FAISS_GIT_TAG="v${FAISS_VERSION}" \
 && echo "Cloning repo using branch/tag: ${FAISS_GIT_TAG}" \
 && git clone -b "${FAISS_GIT_TAG}" --depth 1 \
      -- https://github.com/facebookresearch/faiss.git /opt/faiss

# Apply patch for tag version
COPY patch/${FAISS_VERSION} /opt/faiss

#
# Configure and Build FAISS library and bindings.
#
ARG CUDA_ARCH
ENV FAISS_INSTALL_PREFIX="/opt/faiss/install"
RUN ./configure \
      --with-cuda=/usr/local/cuda \
      --with-cuda-arch="${CUDA_ARCH}" \
      --with-python=python3 \
      --prefix="$FAISS_INSTALL_PREFIX" \
 && make -j$(nproc) \
 && make -C python -j$(nproc)

# Create python wheel for export creating file for copy:
#   /opt/faiss/python/dist/faiss-${FAISS_VERSION}-py3-none-any.whl
RUN cd python \
 && python3 setup.py bdist_wheel

# Uncomment this if you want a local install of faiss for some reason.
# - the PYTHONPATH mod is required for setup.py
#RUN FAISS_SITE="$(python3 -c \"from distutils.sysconfig import get_python_lib; \
#                              print(get_python_lib(prefix=\"${FAISS_INSTALL_PREFIX}\"))\")" \
# && export PYTHONPATH="${FAISS_SITE}:${PYTHONPATH}" \ # && cd /opt/faiss \
# && make install \
# && mkdir -p "$FAISS_SITE" \
# && make -C python install

# Uncomment to perform a full suite of tests
# NOTE: Apparently some tests still require python2.7 (non-function `print`
#       statements).
#RUN apt-get -y update \
# && apt-get -y install curl=7.58.0-2ubuntu3.8 \
# && pip3 install scipy==1.4.1 \
# && make test -j$(nproc)
