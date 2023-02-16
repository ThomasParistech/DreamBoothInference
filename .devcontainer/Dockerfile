# syntax=docker/dockerfile:1.3

# Other version not tested yet
ARG CUDA_VERSION=11.3.0
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CPU_SIZE=4
ARG PYTHON_VERSION=3.10
ARG PYTHON_SUBVERSION=6
ENV PYTHON_FULL_VERSION=${PYTHON_VERSION}.${PYTHON_SUBVERSION}
ARG POTREE_VERSION=1

# Fix Cuda Path
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

ENV DEPENDENCIES="/dependencies"
WORKDIR ${DEPENDENCIES}

RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

# basic packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update                && \
    apt-get -y upgrade               && \
    apt-get install -y                  \
    software-properties-common      \
    sudo                            \
    cmake                           \
    build-essential                 \
    wget                            \
    curl                            \
    git                             \
    swig

#-------------------------------------------------------------
#       Python Installation
#-------------------------------------------------------------
RUN apt-get -y update                && \
    apt-get -y upgrade               && \
    apt-get -y install      \
    build-essential         \
    zlib1g-dev              \
    libncurses5-dev         \
    libgdbm-dev             \
    libnss3-dev             \
    libssl-dev              \
    libreadline-dev         \
    libffi-dev              \
    libsqlite3-dev          \
    wget                    \
    libbz2-dev              \
    liblzma-dev             \
    lzma                    \
    tk-dev

RUN wget https://www.python.org/ftp/python/${PYTHON_FULL_VERSION}/Python-${PYTHON_FULL_VERSION}.tgz     && \
    tar -xf Python-${PYTHON_FULL_VERSION}.tgz                                           && \
    cd Python-${PYTHON_FULL_VERSION}/                                                   && \
    ./configure --enable-optimizations                                  && \
    make -j4                                                            && \
    make install


# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VERSION} get-pip.py

#-------------------------------------------------------------
#       Python3 Module
#-------------------------------------------------------------

# Deps for python modules
RUN apt-get update -y && \
    apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    ffmpeg


#-------------------------------------------------------------
#       Dreambooth
#-------------------------------------------------------------

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==1.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
RUN wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    git+https://github.com/ShivamShrirao/diffusers

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    -U --pre triton

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    accelerate==0.12.0 transformers ftfy bitsandbytes==0.35.0 gradio natsort

RUN --mount=type=cache,target=/root/.cache/pip \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    XFORMERS_DISABLE_FLASH_ATTN=1 \
    pip install \
    -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# Consider loading wheels instead of recompiling xformers
# https://github.com/facebookresearch/xformers/actions/runs/3543179717/jobs/5949406524

#-------------------------------------------------------------
#       Utils
#-------------------------------------------------------------

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    autopep8 pylint mypy fire unimport darglint

#-------------------------------------------------------------
#       Post Processing
#-------------------------------------------------------------

ENV DREAM_BOOTH_INFERENCE="/workspaces/DreamBoothInference"
ENV PYTHONPATH $PYTHONPATH:$DREAM_BOOTH_INFERENCE

## Switch to non-root user
ENV USER=dock
ENV GROUP=sudo

RUN useradd -ms /bin/bash ${USER} && \
    usermod -aG ${GROUP} ${USER}

# Cleaning Part
USER root
RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*

# Resolve authorization problem
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" \
    >/etc/sudoers.d/${USER} && \
    chmod 0440 /etc/sudoers.d/${USER}

# Final User
USER dock
