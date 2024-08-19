# based from https://github.com/horovod/horovod/blob/55a73d4f662bf637f6994a61f62b517ad9554dc1/docker/horovod/Dockerfile
ARG CUDA_DOCKER_VERSION=11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

ARG CUDA_DOCKER_VERSION=11.8.0-devel-ubuntu22.04
ARG CUDNN_VERSION=8.9.7.29-1+cuda11.8
ARG DOCKER_NCCL_VERSION=2.15.5-1+cuda11.8
ARG PYTORCH_VERSION=2.0.1+cu118
ARG TORCHVISION_VERSION=0.15.2+cu118
ARG HOROVOD_VERSION=0.28.1

ARG PYTHON_VERSION=3.10

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

#RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub

RUN echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-12 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn8=${CUDNN_VERSION} \
        libnccl2=${DOCKER_NCCL_VERSION} \
        libnccl-dev=${DOCKER_NCCL_VERSION} \
	nsight-systems-cli \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openjdk-8-jdk-headless \
        openssh-client \
        openssh-server \
        libhwloc-dev \
        libevent-dev \
        libpsm2-dev \
        libfabric-dev \
        libnl-route-3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Open MPI
RUN curl -sfL https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz | tar zx -C /tmp && \
    cd /tmp/openmpi-4.0.3 && ./configure \
        --prefix=/usr/local \
        --with-psm2=/usr \
        --with-libfabric=/usr \
        --enable-mca-no-build=btl-openib \
        --without-verbs \
        --with-hwloc=external \
        --with-libevent=external && \
    make -j all && make install && ldconfig && \
    cd / && rm -rf /tmp/openmpi-4.0.3 && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    denoising_diffusion_pytorch \
    -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    bash -c "HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=0 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 pip install --no-cache-dir -v horovod==${HOROVOD_VERSION}" && \
    horovodrun --check-build && \
    ldconfig

# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.torch as hvd; hvd.init()" && \
    ldconfig
