---
slug: docker-multi-stage
date: 2019-06-21T00:07:25.807Z
title: "Smaller Docker Image using Multi-Stage Build"
description: "Example: CUDA-enabled PyTorch + Apex Image"
tags:
  - deep-learning
  - pytorch
  - docker
keywords:
  - apex
  - pytorch
  - docker
  - deep learning
url: /post/docker-multi-stage-build/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/C2M7DWL2fDk)" >}}

## Why Use Mutli-Stage Build?

Starting from Docker 17.05, users can utilize this new "multi-stage build" feature [[1]]({{<ref "#references" >}}) to simplify their workflow and make the final Docker images smaller. It basically streamlines the "Builder pattern", which means using a "builder" image to build the binary files, and copying those binary files to another runtime/production image.

Despite being an interpreted programming language, many of Python libraries, especially the ones doing scientific computing and machine learning, are built upon pieces written in compiled languages (mostly C/C++). Therefore, the "Builder pattern" can still be applied.

(The following sections assume that you've already read the multi-stage build documentation [[1]]({{<ref "#references" >}}).)

### CUDA-enabled Docker images

We can build CUDA-enabled Docker images using nvidia-docker[[2]]({{<ref "#references" >}}). These types of images can potentially benefit hugely from multi-stage build, because (a) their sizes are quite big (usually multiple GBs) and (b) many packages that use CUDA requires CUDA library to build, but it is only useful in build time.

We are going demonstrate the power of multi-stage build by building a Docker image that has PyTorch 1.1 and NVIDIA Apex[[3]]({{<ref "#references" >}}) installed. Here's a sneak peek of the build image and the final runtime image:

{{< figure src="image-sizes.png" caption="The sizes of the \"build\" image and the final \"runtime\" image. " >}}

The image size has shrunk from **6.88 GB** to **3.82 GB** (~45% reduction).

## Building an PyTorch + NVIDIA Apex Image

**The complete Dockerfile used [can be found here on Github](https://github.com/ceshine/Dockerfiles/blob/master/cuda/pytorch-apex/Dockerfile)**.

### Builder

We are going to use `nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04`[[4]]({{<ref "#references" >}}) as the base image. Note that the `devel` part is required for the optimal NVIDIA Apex build settings. This image already takes 3.1 GB of space.

I personally prefer to use miniconda in Docker images because it's the easiest way to install PyTorch and you get to choose the version of your Python interpreter easily. The following will install `miniconda` into `/opt/conda`:

```
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS build

ARG PYTHON_VERSION=3.7
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

## Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

## Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

Then we install the desired version of Python interpreter, PyTorch from the official instructions, and some other essential scientific computing libraries.

```
RUN conda install -y python=$PYTHON_VERSION && \
    conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch && \
    conda install -y h5py scikit-learn matplotlib seaborn \
    pandas mkl-service cython && \
    conda clean -tipsy
```

Then we install NVIDIA Apex:

```
## Install apex
WORKDIR /tmp/
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

Finally, you can install some packages that are not available on conda via `pip` command.

### Runtime

When you install PyTorch via conda, an independent CUDA binaries are installed. That enables us to use `nvidia/cuda:10.0-base` as the base image. This bare-bone image only uses 115 MB.

Here we create a default user, which can improve security, and copy the whole `/opt/conda` directory into the new runtime image.

```
FROM nvidia/cuda:10.0-base

ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

ENV PATH $CONDA_DIR/bin:$PATH

## Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME

COPY --chown=1000 --from=build /opt/conda/. $CONDA_DIR
```

The new `FROM` marks the start of the second stage.

Unfortunately, it seems the `--chown=1000` has to be hard-coded. Docker cannot handle `--chown=$USERID`. This is a minor inconvenience.

### Building the Dockerfile

Just like any other Dockerfile, build it with something like `docker build -t apex --force-rm .`.

You can use `--target` to control the target stage. This is how I obtain the size of the builder image: `docker build --target build -t build-tmp --force-rm .`.

### Side notes

Copying `/opt/conda` and setting the `$PATH` environment variable might not be the perfect way to migrate the conda installation, but it works well enough for me.

If you intend to have multiple conda environment in one Docker image, this image will fail to execute `conda activate`. You may need to run `conda init bash` for it to work.

Most, if not all, of the reduction in image size come from switching base image. I'm not sure if there are other ways to squeeze more space from the `/opt/conda` folder. If you do, please let me know in the comment section.

## References

1. [Use multi-stage builds | Docker documentation](https://docs.docker.com/develop/develop-images/multistage-build/).
2. [Docker + NVIDIA GPU = nvidia-docker | The Artificial Impostor](https://medium.com/the-artificial-impostor/docker-nvidia-gpu-nvidia-docker-808b23e1657)
3. [NVIDIA/apex - A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch ](https://github.com/NVIDIA/apex)
4. [nvidia/cuda | Docker Hub](https://hub.docker.com/r/nvidia/cuda/)
