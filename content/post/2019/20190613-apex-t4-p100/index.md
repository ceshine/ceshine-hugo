---
slug: apex-t4-p100
date: 2019-06-13T00:07:25.807Z
title: "Mixed Precision Training on Tesla T4 and P100"
description: "Training Wide-Resnet with Apex on Google Colab and Kaggle"
tags:
  - deep-learning
  - pytorch
keywords:
  - apex
  - pytorch
  - colab
  - kaggle
url: /post/apex-t4-p100/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/peach-blossom-landscape-spring-4119429/)" >}}

**tl;dr: the power of Tensor Cores is real. Also, make sure the CPU does not become the bottleneck.**

## Motivation

I've written about Apex in this previous post: [Use NVIDIA Apex for Easy Mixed Precision Training in PyTorch](/post/nvidia_apex/). At that time I only have my GTX 1070 to experiment on. And as we've learned in that post, pre-Volta nVidia cards does not benefit from half-precision arithmetic in terms of speed. It only saves some GPU memory. Therefore, I wasn't able to personally evaluate how much speed boost we can get from mixed precision with Tensor Cores.

Recently, Google Colab starts to allocate [Tesla T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), which has 320 Turing Tensor Cores, with GPU runtime for free. It is a perfect opportunity to do a second run of the previous experiments. (GPU runtimes with K80 GPU are still being allocated, so make sure you have the correct runtime.)

Kaggle also just replaced K80 with P100 in their Kernel offerings. We've mentioned [a source claiming](https://github.com/NVIDIA/apex/issues/76) P100 can benefit from half-precision arithmetic for certain networks. So we're also going to give it a try.

## Experiments

### Setup

- Dataset: Cifar-10
- Batch size 128
- Model: [Wide Resnet](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py)
- 10 epochs
- SGD with momentum
- Linear LR scheduler with Warmup

Github repo: [ceshine/apex_pytorch_cifar_experiment](https://github.com/ceshine/apex_pytorch_cifar_experiment/tree/2019-june-post).

### Google Colab

Notebook snapshots stored in [colab_snapshots](https://github.com/ceshine/apex_pytorch_cifar_experiment/tree/2019-june-post/colab_snapshots) subfolder.

| Level                | GPU | Time        | GPU Memory | Validation Accuracy |
| -------------------- | --- | ----------- | ---------- | ------------------- |
| O0 (Pure FP32)       | T4  | 57min 17s   | 7657 MB    | 86.75%              |
| O1 (Mixed Precision) | T4  | 31min 14s   | 4433 MB    | 85.25%              |
| O2 (Mixed Precision) | T4  | 29min 31s   | 4383 MB    | 88.90%              |
| O3 (Pure FP16)       | T4  | N/A         | 4347 MB    | Not Converged.      |
| O0 (Pure FP32)       | K80 | 1h 43min 7s | 6786 MB    | 88.44%              |

### Kaggle Kernel

Kaggle Kernel used: [APEX Experiment - Cifar 10](https://www.kaggle.com/ceshine/apex-experiment-cifar-10).

| Level                                                                                                    | Time      | GPU Memory | Validation Accuracy |
| -------------------------------------------------------------------------------------------------------- | --------- | ---------- | ------------------- |
| [O0 (Pure FP32)](https://www.kaggle.com/ceshine/apex-experiment-cifar-10?scriptVersionId=15544605)       | 47min 01s | 5677 MB    | 87.49%              |
| [O1 (Mixed Precision)](https://www.kaggle.com/ceshine/apex-experiment-cifar-10?scriptVersionId=15544647) | 47min 34s | 6283 MB    | 88.51%              |
| [O2 (Mixed Precision)](https://www.kaggle.com/ceshine/apex-experiment-cifar-10?scriptVersionId=15556913) | 45min 34s | 5665 MB    | 87.74%              |

### Remarks

1. Since the model was only trained 10 epochs to save time, the validation accuracy does not have any important meanings other than indicating whether the model is converging or not.
1. Training with mixed precision on **T4** is almost **twice as fast** as with single precision, and consumes consistently less GPU memory.
1. Training wide-resnet with mixed precision on **P100** does not have any significant effect in terms of speed. The GPU memory footprints are quite bizarre, though. Theoretically at least O2 level should use much less memory than that.
1. Batch size matters. Because both Kaggle and Colab equip instances with only two weak vCPU, _data preprocessing and loading can quickly becomes the bottleneck_. (When using batch size of 512, training under O0, O1, and O2 cost almost the same time, as most time were spent waiting the CPU.) This problem is much more severe when training smaller models.
