# See https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ARG PYTHON_VERSION=3.7.9

# to prevent potential prompt when running apt get and install during rebuilding.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         vim && \
     rm -rf /var/lib/apt/lists/*


# See also
# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.5.0-cuda10.2-ubuntu18.04/Dockerfile
# https://anaconda.org/pytorch/pytorch/files?version=1.7.1&page=2
# https://medium.com/@zaher88abd/pytorch-with-docker-b791edd67850
# CUDA 10.2-specific steps
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing \
     scikit-learn=0.23.2 numpy=1.19.2 pandas xlrd packaging jsonschema pickleshare seaborn jupyter pip && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 cudatoolkit=10.2 "pytorch=1.7.1=py3.7_cuda10.2.89_cudnn7.6.5_0" && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torchsummary rdt

WORKDIR /workspace
RUN chmod -R a+w .

# create a 'CTGAN' folder and copy files into this folder.
WORKDIR /CTGAN
COPY . .

ENV PYTHONPATH="/CTGAN"
CMD ["python", "main.py"]

# To save docker image locally and transfer via scp, see reply by JSON C11 in https://stackoverflow.com/questions/24482822/how-to-share-my-docker-image-without-using-the-docker-hub
# docker save -o <path for created tar file> <image name>
# docker load -i <path to docker image tar file>
