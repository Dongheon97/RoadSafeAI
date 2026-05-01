FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/MS3D:/MS3D/tracker \
    DISPLAY=:1 \
    QT_X11_NO_MITSHM=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        htop \
        libdbus-1-3 \
        libfontconfig1 \
        libfreetype6 \
        libgl1 \
        libglib2.0-0 \
        libglu1-mesa \
        libsm6 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-render0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxext6 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libxrender-dev \
        python3-distutils \
        python3.8 \
        python3.8-dev \
        tree \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    wget -q https://bootstrap.pypa.io/pip/3.8/get-pip.py -O /tmp/get-pip.py && \
    python /tmp/get-pip.py && \
    rm /tmp/get-pip.py

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
        numpy==1.23.5 \
        llvmlite==0.39.1 \
        numba==0.56.4 \
        torch==1.8.1+cu111 \
        torchvision==0.9.1+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html

RUN python -m pip install \
        ConfigArgParse==1.7.5 \
        GitPython==3.1.46 \
        SharedArray==3.2.4 \
        ccimport==0.3.7 \
        easydict==1.13 \
        filterpy==1.4.5 \
        fire==0.7.1 \
        imageio==2.35.1 \
        matplotlib==3.7.5 \
        ninja==1.13.0 \
        open3d==0.18.0 \
        opencv-python-headless==4.8.1.78 \
        pandas==2.0.3 \
        pccm==0.3.4 \
        pillow==10.4.0 \
        portalocker==3.0.0 \
        pybind11==3.0.2 \
        pyquaternion==0.9.9 \
        pyyaml==6.0.3 \
        scikit-image==0.21.0 \
        scikit-learn==1.3.2 \
        scipy==1.10.1 \
        shapely==2.0.7 \
        tensorboardX==2.6.2.2 \
        tqdm==4.67.3 \
        virtualenv==21.2.0 \
        wandb==0.16.6

RUN python -m pip install spconv-cu111==2.1.25

WORKDIR /MS3D

COPY . /MS3D

RUN git config --global --add safe.directory /MS3D && \
    chmod +x /MS3D/docker_entrypoint.sh && \
    python setup.py build_ext --inplace && \
    python -m pip install -e . --no-build-isolation

ENTRYPOINT ["/MS3D/docker_entrypoint.sh"]
