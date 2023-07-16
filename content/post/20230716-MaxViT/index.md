---
slug: MaxViT
date: 2023-07-16T00:00:00.000Z
title: "[Notes] MaxViT: Multi-Axis Vision Transformer"
description: "Multi-axis attention to enable both local and global interactions efficiently"
tags:
  - python
  - pytorch
  - cv
  - deep_learning
keywords:
  - python
  - pytorch
  - cv
  - deep learning
url: /post/MaxViT/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://www.pexels.com/photo/crop-woman-making-schedule-in-planner-5239797/)" >}}

[MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)(1) is a paper jointly produced by Google Research and University of Texas at Austin in 2022. The paper proposes a new attention model, named multi-axis attention, which comprises a blocked local and a dilated global attention module. In addition, the paper introduces MaxViT architecture that combines multi-axis attentions with convolutions, which is highly effective in ImageNet benchmarks and downstream tasks.

## Multi-Axis Attention

<div style="max-width: 700px; margin-left: auto; margin-right: auto;">{{< figure src="swin-attention.png" caption="Source: [2]" >}}</div>

The Swin Transformer[2] uses local shifted non-overlapping windows to restrict attention, which reduces the computational complexity and improves performance on tasks that require local information. However, the window-based attention has limited capacity due to the loss of non-locality, making it difficult to scale on larger datasets like ImageNet-21K and JFT.

<div style="max-width: 700px; margin-left: auto; margin-right: auto;">{{< figure src="Max-SA-1.png" caption="Multi-axis self-attention (Max-SA) [1]" >}}</div>

The first part of Max-SA is **block attention**, which involves using a local attention window of a fixed size to facilitate local interactions. This is similar to Swin attention, but without shifting windows. It reshapes the input tensor of shape `$(H, W, C)$` to `$(\frac{H}{P} \times \frac{W}{P},P \times P, C)$` and applies attention on the second dimension (`$P \times P$`) (P is the window size).

The second part of Max-SA is what makes it unique (and also confusing). It is the **grid attention**, which performs sparse global attention. The input tensor is reshaped to `$(G \times G, \frac{H}{G} \times \frac{W}{G}, C)$`, but this time, the attention is applied on the **first dimension** (`$G \times G$`) (G is the grid size). The size of attention window is therefore also fixed, making both the compute complexity of both the block attention and grid attention linear to the input size.

(Throughout this article, I use the terms "pixel" and "element" interchangeably to denote a single element in the 2D space.)

The figure above illustrates an example of applying Max-SA on a 2D input. The window size (P) and grid size (G) are both 4. The size of the input tensor is `$(8, 8, 1)$`. The block attention split the tensor into 4 blocks of 4 by 4 windows (`$(2 \times 2, 4 \times 4, 1)$`), and the attention is applied within elements of the same color. The grid attention interlaces `$\frac{8}{4} \times \frac{8}{4} = 4$` 4 by 4 grids on the input tensor,and the attention is applied within each 4 by 4 grid.

The paper refers to the local `$\frac{H}{G} \times \frac{W}{G}` local tensors as "windows" of adaptive size, which can be confusing since the elements within a window do not interact with each other in the grid attention. Rather, a "window" here simply means to the size of the tensor that contains exactly one element from each grid.

### On the "globalness" of Max-SA

Although Max-SA has a _global_ grid attention, its receptive field for the entire image is not guaranteed. For example, if the input size is `$(18, 18, 1)$`, the block size is 3, and the grid size is 3, then each element of the Max-SA attention's output can only see `$(9, 9, 1)$` pixels or elements. The input tensor is partitioned into `$6 \times 6 = 36$` blocks of 3 by 3 windows in the block attention. The grid attention divides the windows into two groups, each with 18 windows, and information can only be exchanged within these groups.

The MaxViT architecture utilizes downsampling stages and convolutions to increase the receptive field to the entire image. However, when used independently, it's important to note that the Max-SA attention is not always global, unlike the Swin attention that can exchange information with neighboring windows.

The grid attention allows attending to far-away pixels without increasing the compute complexity, but it doesn't attend to the entire image. It's not recommended to stack multiple layers of Max-SA attentions with the same hyper-parameters because the receptive field may not increase.

## MaxViT Architecture

<div style="max-width: 700px; margin-left: auto; margin-right: auto;">{{< figure src="MaxViT-1.png" caption="MaxViT [1]" >}}</div>

The MaxViT Architecture is a hybrid of convolution and multi-axis attention. It begins by passing the input image through a convolutional stem. The output of the stem then passes through a series of Max-SA blocks, which is a MBConv block with a squeeze-and-excitation (SE) module followed by a Max-SA module. The overall architecture is comparable to conventional convolutional architecture like ResNet, but with the convolutional blocks substituted by Max-SA blocks.

<div style="max-width: 700px; margin-left: auto; margin-right: auto;">{{< figure src="MaxViT-2.png" caption="MaxViT Performance [1]" >}}</div>

MaxVit differs from CoAtNet[3], a competitive hybrid architecture of convolution and attention, in that it has a simpler design that repeats the same block throughout the network. In contrast, CoAtNet uses different designs in each stage. Both architectures use a relative attention that is input adaptive and translation equivariant. However, CoAtNet does not use window-based attention and its attention window size is not fixed.

## References

1. Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., & Li, Y. (2022). [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697).
2. Liu, Ze, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. (2021). [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).
3. Dai, Z., Liu, H., Le, Q. V., & Tan, M. (2021). [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803).
