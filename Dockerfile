## Docker

## Unofficial Dockerfile for 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
## https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Use the base image with PyTorch and CUDA support
# FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# NOTE:
# Building the libraries for this repository requires cuda *DURING BUILD PHASE*, therefore:
# - The default-runtime for container should be set to "nvidia" in the deamon.json file. See this: https://github.com/NVIDIA/nvidia-docker/issues/1033
# - For the above to work, the nvidia-container-runtime should be installed in your host. Tested with version 1.14.0-rc.2
# - Make sure NVIDIA's drivers are updated in the host machine. Tested with 525.125.06
ENV DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

RUN apt-get update && \
    apt-get install -y wget

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH


COPY  environment.yml /tmp/environment.yml
WORKDIR /tmp/

RUN conda env create -f environment.yml && \
    conda init bash && exec bash

# Install torch afterwards
RUN conda run -n mega_sam pip install torch==2.0.1+cu118 torchvision==0.15.2 -f https://download.pytorch.org/whl/torch_stable.html

# installing torch scatter afterwards
RUN conda run -n mega_sam python -m pip install --no-cache-dir \
  torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html


RUN wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
RUN conda install -n mega_sam xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2


COPY base /tmp/base/
WORKDIR /tmp/base/

# g++ for installing afterwards
RUN apt-get update && apt-get install -y \
    build-essential ffmpeg

RUN conda run -n mega_sam python setup.py install


# For visualization only
RUN conda run -n mega_sam pip install viser

WORKDIR /mega_sam
ENV PYTHONPATH="/mega_sam/UniDepth:$PYTHONPATH"

# This error occurs because there’s a conflict between the threading layer used
# by Intel MKL (Math Kernel Library) and the libgomp library, 
# which is typically used by OpenMP for parallel processing. 
# This often happens when libraries like NumPy or SciPy are used in combination
# with a multithreaded application (e.g., your Docker container or Python environment).
# Solution, set threading layer explicitly! (GNU or INTEL)
ENV MKL_THREADING_LAYER=GNU