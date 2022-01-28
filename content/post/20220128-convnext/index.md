---
slug: convnext-notes
date: 2022-01-28T00:00:00.000Z
title: "[Notes] Understanding ConvNeXt"
description: "A ConvNet for the 2020s"
tags:
  - python
  - pytorch
  - cv
keywords:
  - python
  - pytorch
  - cv
  - deep learning
  - research
url: /post/convnext-notes/
---

{{< figure src="featuredImage.jpg" caption="[credit](https://unsplash.com/photos/Oh15EWXAMvI)" >}}

## Introduction

Hierarchical Transformers (e.g., Swin Transformers[1]) has made Transformers highly competitive as a generic vision backbone and in a wide variety of vision tasks. A new paper from Facebook AI Research — “A ConvNet for the 2020s”[2] — gradually and systematically “modernizes” a standard ResNet[3] toward the design of a vision Transformer. The result is a family of pure ConvNet models dubbed **ConvNeXt** that compete favorably with Transformers in terms of accuracy and scalability.

Because ConvNeXt is borrowing designs from vision Transformers, the paper shows us how much progress in the vision Transformers can be explained by superior architectural design. And the evidence so far indicated that maybe self-attention (unique in Transformers) by itself is not enough to dominate all computer vision tasks. Many of the new techniques introduced in vision Transformers that pushed the limit can also be applied to pure convolution networks.

Personally, I feel particularly excited about this paper because it successfully replaced batch normalization with layer normalization without any loss of accuracy. “Big Transfer (BiT)”[4] similarly replaced batch normalization with group normalization and weight standardization in 2019. Unfortunately, it is not widely adopted and most new ConvNet papers still use batch normalization in their released pretrained models. Hopefully, this layer normalization can stand the test of time and we can no longer be limited by the batch normalization for its degraded performance in small batch sizes.

## The “Modernization” Process

<div style="max-width: 450px; margin-left: auto; margin-right: auto;">{{< figure src="fig-2.png" caption="Source: [2]" >}}</div>

(All models are trained and evaluated on ImageNet-1K.)

### Training Techniques

1. Number of epochs: **90** -> **300**
2. Optimizer: **AdamW**
3. Augmentation techniques:
   - Mixup and Cutmix. [A coin flip is used to decide which one to use for the batch](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L132). [The Mixup implementation from timm is used](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L273).
   - Label smoothing. The [smoothing factor](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L105) is given as [a parameter to the Mixup function](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L273) and the [LabelSmoothCrossEntropy loss will be used](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L373).
   - RandAugment and random erasing: the `create_transform` [helper function from timm](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/datasets.py#L58) is used. The default random erasing probability [is 25%](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L114). The default RandAugment strategy [is rand-m9-mstd0.5-inc1](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/main.py#L103).
   - Stochastic depth: [DropPath from timm is used](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/models/convnext.py#L35).

### Macro Design

<div style="max-width: 450px; margin-left: auto; margin-right: auto;">{{< figure src="table-9.png" caption="Source: [2]" >}}</div>

- Stage compute ratio: the number of blocks in each stage goes from `(3, 4, 6, 3)` to `(3, 3, 9, 3)` (see Table 9) to make the compute ratio `1:1:3:1`.
- Patchifying stem: a 7x7 convolution with stride 2 followed by a 3x3 max pooling with stride 2 -> a 4x4 convolution with stride 4 (from overlapping to non-overlapping).
- ResNeXt-ify: use depthwise convolution and increase the network width.
- Inverted Bottleneck: instead of 384 -> 96 -> 384, the network widths (number of channels) becomes 96 -> 384 -> 96 ((a) to (b) in Fig 3). (_Note that the number of channels in the last block of (b) is wrong. It should be 384 -> 96._)
- Larger kernel size: use a 7x7 convolution instead 3x3 and move the convolution layer up ((b) to (c) in Fig 3).

<div style="max-width: 450px; margin-left: auto; margin-right: auto;">{{< figure src="fig-3.png" caption="Source: [2]" >}}</div>

### Micro Design

<div style="max-width: 450px; margin-left: auto; margin-right: auto;">{{< figure src="fig-4.png" caption="Source: [2]" >}}</div>

- Replacing ReLU with GELU
- Fewer activation function: from three to only one between the two 1x1 layers. This has been proven to be very effective (a 0.7% boost).
- Fewer normalization layers: from three to only one between the 7x7 and the 1x1 layers.
- Substituting BN with LN: the Layer Normalization version perform slightly better.
- Separate downsampling layers: the downsampling layer consists of [a Layer Normalization and a 2x2 convolution with stride 2](https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/models/convnext.py#L79). They are added between stages.

## Performance

<div style="max-width: 450px; margin-left: auto; margin-right: auto;">{{< figure src="table-1.png" caption="Source: [2]" >}}</div>

Depthwise convolutions reduce the computation comparing to regular convolutions, but require the similar amount of memory to process the data. The performance of a model that heavily uses depthwise convolutions can become bound by memory bandwidth instead of computation capability. It is the main reason behind the slow speed of EfficientNet models[5] despite of their significantly lower FLOPs.

The ConvNeXt models also use depthwise convolution extensively, so it is very nice to see that the paper compares the throughputs of ConvNext models and other popular models including RegNet, EfficientNet, DeiT, and Swin Transformers. Table 1 shows that ConvNeXt is both performant and accurate.

## Conclusion

This paper is clearly written and very informative. I particularly like the way they design and present the modernization process. Each step is well explained and justified. The results are no doubt very impressive. I can't wait to try ConvNeXt on my datasets and see how it performs.

## Reference

1. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., … Guo, B. (2021). [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](http://arxiv.org/abs/2103.14030).
2. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). [A ConvNet for the 2020s](http://arxiv.org/abs/2201.03545).
3. He, Zhang, Ren, & Sun. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). In Conference on Computer Vision and Pattern Recognition, 2016.
4. Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., & Houlsby, N. (2019). [Big Transfer (BiT): General Visual Representation Learning](http://arxiv.org/abs/1912.11370).
5. Tan, M., & Le, Q. V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](http://arxiv.org/abs/1905.11946).
