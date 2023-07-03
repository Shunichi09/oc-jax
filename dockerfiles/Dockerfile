FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install essential
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    sudo \
    libopencv-dev \
    curl \
    g++ \
    make \
    git \
    snapd \
    zip \
    unzip \
    wget \
    libssl-dev \
    freeglut3-dev \
    build-essential \
    software-properties-common \
    lsb-core \
    ffmpeg \
    apt-utils

# GPU environment variable
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}display,compute

# Install python3 and venv
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python3-tk \
    python3-venv \
    python3-dev \
    libffi-dev

# Install other dependencies
RUN apt-get update -y && apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6

# Add sudo user
ARG USER
ARG USER_ID
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -u $USER_ID -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd

RUN echo "Defaults visiblepw"             >> /etc/sudoers
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install venv
ENV VENV_PATH "/home/${USER}/venv/venv_apop"
WORKDIR ${VENV_PATH}
RUN python3 -m venv ${VENV_PATH}

# Upgrade basics
RUN ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel

# Install required python packages
RUN mkdir -p /home/${USER}/tmp/deps
COPY ./requirements.txt /home/${USER}/tmp/deps
RUN ${VENV_PATH}/bin/pip install -U -r /home/${USER}/tmp/deps/requirements.txt

RUN echo "alias load_apop_venv='source ${VENV_PATH}/bin/activate'" >> /home/${USER}/.bashrc

# Remove apt caches
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /home/$USER

# Install Mujoco
RUN mkdir -p .mujoco
WORKDIR /home/$USER/.mujoco
ENV MUJOCO_VERSION "2.3.6"
ENV MUJOCO_ARCHIVE "mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
RUN wget -q "https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/${MUJOCO_ARCHIVE}"
RUN wget -q "https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/${MUJOCO_ARCHIVE}.sha256"
RUN shasum -c "${MUJOCO_ARCHIVE}.sha256"
RUN tar -xvzf ${MUJOCO_ARCHIVE}

# chown
RUN chown -R $USER /home/$USER

# Switch user
USER $USER

CMD [ "bash" ]
