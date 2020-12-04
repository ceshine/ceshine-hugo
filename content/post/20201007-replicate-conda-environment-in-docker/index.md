---
slug: replicate-conda-environment-in-docker
date: 2020-10-07T00:00:00.000Z
title: "Replicate Conda Environment in Docker"
description: "A quicker way to share your models"
tags:
  - tip
  - docker
keywords:
  - tip
  - docker
  - conda
url: /post/replicate-conda-environment-in-docker/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/VULVydw3nV0)" >}}

## Introduction

You just finished developing your prototype in a Conda environment, and you are eager to share it with stakeholders, who may not have the required knowledge to recreate the environment to run your model on their end. Docker is a great tool that can help in this kind of scenario (p.s: [it can utilize GPU via nvidia-docker](https://medium.com/the-artificial-impostor/docker-nvidia-gpu-nvidia-docker-808b23e1657)). Just create a Docker image and share it with the stakeholders, and your model will run on their device the same way it runs on yours.

To create a Docker image, you have to replicate the Conda environment you used to develop the model inside Docker. I used to start by writing down packages from memory, and then add the missing ones one-by-one using trial-and-errors. It is very time-consuming and borderline stupid, now that I've learned a much better way to do it.

## Instructions

### Export Conda Environment

This command writes all the packages in your environment to a YAML file[1]:

```bash
conda env export > environment.yml
```

Alternatively, you can only export packages you specifically chose using the `--from-history` flag:

```bash
conda env export --from-history > environment.yml
```

However, this command will ignore packages installed via pip, and I've found that the `channels` field often omits some channels. A better way to utilize this is to use this as a reference to trim down the YAML from the first command.

### Create Dockerfile

Here we use a `cuda-base` base image from Nvidia, so we can use the GPU (you don't usually need `-runtime` or `-devel` image if you're using PyTorch since Conda will install a copy of CUDA in your environment when you install PyTorch):

```dockerfile
FROM nvidia/cuda:10.2-base-ubuntu18.04
```

First, you need to install `miniconda` (a minimal installer for Conda):

```dockerfile
## Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*
```

It's recommended to install mamba to speed up Conda (h/t to [Jeremy Howard](https://twitter.com/jeremyphoward/status/1305342912356478977)):

```dockerfile
RUN conda install -y mamba -c conda-forge
```

Here comes the main dish — copy the YAML file into the Docker image, and then update the base/root Conda environment with it:

```dockerfile
ADD ./environment.yml .
RUN mamba env update --file ./environment.yml &&\
    conda clean -tipy
```

You can also replicate the environment in a separate environment using `mamba env create -f environment.yml`. However, switching environment inside Docker containers can be very cumbersome. And there's really no good reason for having multiple environments inside a Docker image. It kind of defeats the purpose of containerization.

And voalá! The base Conda environment in the Docker image is now exactly the same as your environment. You can now move your code and models into the image and they should run without any problems.

### A Complete Example

{{< gist ceshine 77623d9972c2369bf0ffd40068792caf >}}

## Reference

- [Managing environments - conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
