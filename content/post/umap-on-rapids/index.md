---
slug: umap-on-rapids-15x-speedup
date: 2019-03-30T07:16:26.582Z
author: ""
title: "UMAP on RAPIDS (15x Speedup)"
description: "Exploring the new GPU acceleration library from NVIDIA"
images:
  - featuredImage.jpeg
  - 1*mqmwjXeCosyF6Lt4uAo7_A.png
  - 1*SF1AqkGPQG2gVnr5pmTKvA.png
  - 1*Hta9J-2Vy0GyA_jSgGHGWg.png
tags:
  - tools
  - dataviz
  - rapids
  - docker
  - python
keywords:
  - machine-learning
  - data-visualization
  - data-science
url: /post/umap-on-rapids-15x-speedup/
---

{{< figure src="featuredImage.jpeg" caption="[A_Different_Perspective](https://pixabay.com/users/A_Different_Perspective-2135817/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3648832) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3648832)" >}}

**Exploring the new GPU acceleration library from NVIDIA**

# RAPIDS

[RAPIDS](https://rapids.ai/) is a collection of Python libraries from NVIDIA that enables the users to do their data science pipelines entirely on GPUs. The two main components are `cuDF` and `cuML`. The `cuDR` library provides Pandas-like data frames, and `cuML` mimics `scikit-learn`. There’s also a `cuGRAPH` graph analytics library that have been introduced in the latest release ([0.6 on March 28](https://medium.com/rapids-ai/the-road-to-1-0-building-for-the-long-haul-657ae1afdfd6)).

> The RAPIDS suite of open source software libraries gives you the freedom to execute end-to-end data science and analytics pipelines entirely on GPUs. RAPIDS is incubated by [NVIDIA®](https://nvidia.com) based on years of accelerated data science experience. RAPIDS relies on [NVIDIA CUDA®](https://developer.nvidia.com/cuda-toolkit) primitives for low-level compute optimization, and exposes GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

This NeurIPS 2018 talk provides an overview and vision of RAPIDS when it was launched:

{{< youtube k9E-YSWQxIU >}}

The RAPIDS project also provides [some example notebook](https://github.com/rapidsai/notebooks). Most prominently [the cuDF + Dask-XGBoost Mortgage example](https://github.com/rapidsai/notebooks/blob/branch-0.6/mortgage/E2E.ipynb). (The link to the dataset in the notebook is broken. [Here’s the correct one](https://rapidsai.github.io/datasets/).)

# Does RAPIDS Help in Smaller Scales?

The mortgage datasets are huge. The 17-year one takes 195 GB, and the smallest 1-year one takes 3.9 GB. The parallel processing advantage of GPUs is obvious for larger datasets. I wonder if smaller datasets can benefit from RAPIDS as well?

In the following example, we use the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) (~30MB) and run [UMAP (Uniform Manifold Approximation and Projection)](https://umap-learn.readthedocs.io/en/latest/) algorithm for dimension reduction and visualization. We compare the performances of the [CPU implementation](https://github.com/lmcinnes/umap) and the [RAPIDS implementation](https://rapidsai.github.io/projects/cuml/en/0.6.0/api.html#umap).

It turns out the RAPIDS implementation can be **15x times faster** (caveat: I only ran the experiments **two times**, but they returned roughly the same results):

{{< figure src="1*mqmwjXeCosyF6Lt4uAo7_A.png" >}}

The visualization shows the end products from CPU and RAPIDS implementation are very similar.

{{< figure src="1*SF1AqkGPQG2gVnr5pmTKvA.png" caption="*From the CPU implementation.*" >}}

{{< figure src="1*Hta9J-2Vy0GyA_jSgGHGWg.png" caption="*From the RAPIDS implementation.*" >}}

The RAPIDS implementation lacks some features, though. For example, the random state cannot be set, and the distance metric is fixed (it is not stated in the documentation. I assume it is Euclidean.). But if you don’t need those features, RAPIDS can save you a lot of time.

## Source Code and Environment Setup

I used the [official Docker image](https://docs.rapids.ai/containers/rapids-demo) (*rapidsai/rapidsai:0.6-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7*) to run the following notebook.

{{< gist ceshine f0a09fa24ddc10cef4ddc2a41b18e53d >}}

To pull the image and start an container, run:

```
$ docker pull `rapidsai/rapidsai:0.6-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7`
$ docker run --runtime=nvidia \
        --rm -it \
        -p 8888:8888 \
        -p 8787:8787 \
        -p 8786:8786 \
        `rapidsai/rapidsai:0.6-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7`
(rapids) root@container:/rapids/notebooks# bash utils/start-jupyter.sh`
```


# Fin

Thanks for reading! This is a short post introducing RAPIDS and presents a simple experiments showing how RAPIDS can help you speed up the UMAP algorithm. RAPIDS is still a very young project, and very new to me. I’ll try to use RAPIDS in my future data analytics projects and maybe write more posts about it. Stay tuned!