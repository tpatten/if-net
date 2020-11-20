FROM nvidia/cuda:10.2-base-ubuntu18.04

ENV TZ=Europe/Vienna
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y sudo wget bash zip git rsync build-essential software-properties-common ca-certificates xvfb vim

RUN apt-get install -y python3.7-venv python3.7-dev python3-pip

RUN apt-get install -y libsm6 libxrender1 libfontconfig1 libpython3.7-dev libopenblas-dev

RUN apt-get install -y meshlab

RUN python3.7 -m pip install numpy==1.16.6 \
                             scipy \
                             matplotlib \
                             tensorboard==1.14.0 \
                             scikit-learn \
                             cython==0.29.15 \
                             trimesh==3.6.18 \
							 scikit-image==0.14.2 \
							 pykdtree==1.3.1 \
							 future==0.18.2 \
							 pandas==1.0.3 \
                             pymcubes==0.1.0 \
                             tqdm \
                             open3d

# RUN python3.7 -m pip install cudatoolkit==9.0

RUN python3.7 -m pip install torch==1.2.0 torchvision==0.4.0

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
