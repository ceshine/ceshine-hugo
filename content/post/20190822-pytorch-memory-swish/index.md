---
slug: pytorch-memory-swish
date: 2019-08-22T00:00:00.000Z
title: "More Memory-Efficient Swish Activation Function"
description: "And How to Profile PyTorch GPU Memory Usage"
tags:
  - pytorch
  - tips
keywords:
  - pytorch
url: /post/pytorch-memory-swish/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/people-friends-together-happy-4050698/)" >}}

# Motivation

Recently I've been trying out [_EfficientNet_ models implemented in PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch). I've managed to successfully fine-tune pretrained EfficientNet models on my data set and reach accuracy on par with the mainstream ones like _SE-ResNeXt-50_. However, training the model from scratch has proven to be much harder.

Fine-tuned _EfficientNet_ models can reach the same accuracy with much smaller number of parameters, but they seem to occupy a lot of GPU memory than it probably should (comparing to the mainstream ones). There is an open issue on the Github Repository about this problem — [[lukemelas/EfficientNet-PyTorch] Memory Issues](https://github.com/lukemelas/EfficientNet-PyTorch/issues/18).

Github user [@selina](https://github.com/seilna) suggested that the batch normalization and [Swish activation](https://arxiv.org/abs/1710.05941) are the bottlenecks, and claming that by using [**_custom ops_** in PyTorch](https://pytorch.org/docs/stable/notes/extending.html), we can reduce GPU memory usage by up to 30%.

## Custom Swish Function

This is the most straightforward implementation of a Swish activation module used in _EfficientNet_ ($f(x) = x \cdot \sigma(\beta x)$ with $\beta = 1$):

{{< gist ceshine 8d843a60eb6e20189dd805851517172a >}}

The gradients of this module are handled automatically by PyTorch.

This is the Swish activation module implemented using [custom ops](https://pytorch.org/docs/stable/notes/extending.html):

{{< gist ceshine 4edf541793907864339d55c112309019 >}}

Here we handle the gradients explicitly. We keep a copy of the input tensor and use it in back-propagation stage to calculate the gradients.

**Does this latter version really significantly reduce the GPU memory footprint?** — This is the question we want to answer in the following section.

# Profiling CUDA Memory Usage

I've just learned that now PyTorch has a handy function `torch.cuda.memory_allocated()` [that can be used to profile GPU memory usage](https://pytorch.org/docs/stable/cuda.html#memory-management):

> Returns the current GPU memory occupied by tensors in bytes for a given device.
>
> _Note: This is likely less than the amount shown in nvidia-smi since some unused memory can be held by the caching allocator and some context needs to be created on GPU. See Memory management for more details about GPU memory management._

I used to extract that information from calls to `nvidia-smi` command, but the number reported by `nvidia-smi` will be very inaccurate due to the PyTorch memory caching mechanism.

A simple fully-connected neural network was created to be tested against:

{{< gist ceshine 208e7c106d060e416352378b2fd5d24b >}}

We simply inserted `torch.cuda.memory_allocated()` between model training statements to measure GPU memory usage. For more sophisticated profiling, you should check out something like [pytorch-memlab](https://pypi.org/project/pytorch-memlab/).

{{< gist ceshine 6a3bd70506e021dd9392fc7e312bfe96 >}}

## Observations

When using batch sizes of 128, the GPU memory footprints of the training loop were:

```
(1st step)
data: 524 MB
forw: 1552 MB
loss: 1552 MB
back: 1044 MB
step: 1044 MB
====================
(2nd step)
data: 1044 MB
forw: 2072 MB
loss: 2072 MB
back: 1044 MB
step: 1044 MB
(The latter steps are exactly the same as the second one.)
```

The difference between the first and the latter steps is probably due to gradients not being allocated until `loss.backward()` is called.

The peak memory usage happens right after the forward-propagation. As has been shown in the custom-op implementation of Swish, **some function requires PyTorch to save some forms of the input tensors to be able to back-propagate**. Those saved information are discarded after the backward phase. By this logic, we can guess that training with larger batch sizes will use more memory, and this intuition is confirmed by experiments:

{{< figure src="table.png" caption="Peak Memory Usage" >}}

The custom-op version of Swish uses **almost 20%** less memory when batch size is 512. PyTorch augograd probably decides to save more information in the forward phase to avoid some re-calculation in the backward phase. Note that in the custom-op version, `i * (1 - sigmoid_i)` in the backward function can be refactored to reuse the calculated number `i * torch.sigmoid(i)` in the forward function.

**The custom-op version might have traded some speed for memory**. I have not done any profiling on time yet. But as the bottleneck in my system is often the GPU memory, I'd happily accept the tradeoff anyway.

# Source Code

[My fork of _EfficientNet-PyTorch_](https://github.com/ceshine/EfficientNet-PyTorch) has replaced the original swish function with the more memory-efficient one.

The notebook used to run the experiments:

{{< gist ceshine b05b77dd31c407bdf4577227ece442bf >}}
