FROM ghcr.io/yamaura/denoising-diffusion-pytorch-horovod
ARG NCCL_TESTS_VERSION=v2.13.10

RUN git clone https://github.com/NVIDIA/nccl-tests.git /nccl-tests -b $NCCL_TESTS_VERSION && \
    cd /nccl-tests && make MPI=1 -j
WORKDIR /nccl-tests
