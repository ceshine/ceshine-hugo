---
slug: xcit-part-1
date: 2021-07-24T00:00:00.000Z
title: "[Notes] Understanding XCiT - Part 1"
description: "Cross-Covariance Attention(XCA) Block"
tags:
  - python
  - pytorch
  - transformers
  - cv
keywords:
  - python
  - pytorch
  - transformers
  - cv
  - deep learning
url: /post/xcit-part-1/
---

{{< figure src="featuredImage.jpeg" caption="[credit](https://unsplash.com/photos/godmBw_gLDg)" >}}

## Overview

XCiT: [Cross-Covariance Image Transformers](http://arxiv.org/abs/2106.09681)[1] is a paper from Facebook AI that proposes a “transposed” version of self-attention that operates across feature channels rather than tokens. This cross-covariance attention has linear complexity in the number of tokens (the original self-attention has quadratic complexity). When used on images as in vision transformers, this linear complexity allows the model to process images of higher resolutions and split the images into smaller patches, which are both shown to improve performance.

{{< figure src="figure-1.png" caption="[from [1]](http://arxiv.org/abs/2106.09681)" >}}

XCiT replaces the self-attention layer in vision transformers with a Cross-Covariance Attention(XCA) block and a Local Patch Interaction(LPI) block. The XCA block can be understood as a “dynamic” `1×1` convolution layer, as the convolution filters depend on the input data. The LPI block is simply depth-wise `3×3` convolution layers.

There is also a class attention layer[2] when XCiT is trained for image classification.

We're going to review the implementation details of the XCA block, the LPI block, and the class attention layer by going through the code from [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/xcit.py) (which is mostly the same as the [official implementation](https://github.com/facebookresearch/xcit/blob/master/xcit.py) of XCiT). The XCA block will be covered in this part 1 post. The rest will be in part 2.

## Cross-Covariance Attention(XCA) block

The key($K=XW_k$) and query($Q=XW_q$) are both `N×C` matrices as in the vanilla transformer. However, instead of calculating the similarity between tokens ($QK^T$, an `N×N` matrix), the XCA calculates the similarities between channels/features ($K^TQ$, a `C×C` matrix). The latter is also known as the (unnormalized) [cross-covariance matrix](https://www.wikiwand.com/en/Cross-covariance_matrix) between `K` and `Q` (the notation is a slightly different in the linked Wikipedia page because `X` and `Y` in that page are column vectors). That's where the name XCA comes from.

Now we move on to review the [XCA implementation in PyTorch](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/xcit.py#L251). Let's take a look at the complete module. It's okay if you feel overwhelmed. We'll go through the implementation line by line in just a moment.

{{< gist ceshine 9b38a7fa92811fc2733339bf50ec4907 >}}

The first line in the `forward` method extracts the shape parameters from the input batch (a `B×N×C` tensor. B is the batch size; N is the token length; C is the channel/filter size):

```python
B, N, C = x.shape
```

The second line computes the `Q`, `K`, `V` matrices. Similar to the vanilla transformer, these matrices are splits into equal-sized “heads” as it is empirically shown to improve convergence as well as performance:

```python
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
```

The `qkv` is a linear layer — `nn.Linear(dim, dim * 3, bias=qkv_bias)`. The result of `self.qkv(x)` is a `B×N×3C` tensor. It is then reshaped to `B×N×3xHx(C/H)` and permuted to `3xB×Hx(C/H)xN`.

The following line simply split the previous tensor into three `B×Hx(C/H)xN` tensors:

```python
q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
```

The `Q` and `K` matrices are $l2$-normalized. The paper claims that normalizing these two matrices “strongly enhanced the stability of training, especially when trained with a variable number of tokens.” Note that the normalization happens in the token dimension:

```python
q = torch.nn.functional.normalize(q, dim=-1)
k = torch.nn.functional.normalize(k, dim=-1)
```

The following line calculates the attention matrices with a learnable temperature $K^TQ/\tau$:

```python
attn = (q @ k.transpose(-2, -1)) * self.temperature
attn = attn.softmax(dim=-1)
attn = self.attn_drop(attn)
```

`q @ k.transpose(-2, -1)` creates a `B×hx(C/H)x(C/H)` tensor. A Softmax is applied to the last dimension to create the attention vectors (in this case, how much attention a channel from the query should pay to another channel from the key). Then a dropout is applied to the tensors.

Now the attention matrices are used as the filters of `1×1` convolutions on the value matrix:

```python
x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
```

`attn @ v` creates a `B×Hx(C/H)xN` tensor. The tensor is then permuted to `BxNxHx(C/H)` and then reshaped to `BxNxC`. Now we're back to the original shape!

Finally, a linear projection is applied on the tensor (with dropout) to mix the information from all heads:

```python
x = self.proj(x)
x = self.proj_drop(x)
```

That's it! This is how the XCA block works. I hope this walk-through helps you better understand what's happening under the hood. The LPI block and class attention layer will be covered in part 2. Please stay tuned.

## References

1. El-Nouby, A., Touvron, H., Caron, M., Bojanowski, P., Douze, M., Joulin, A., … Jegou, H. (2021). [XCiT: Cross-Covariance Image Transformers.](http://arxiv.org/abs/2106.09681)
2. Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., & Jégou, H. (2021). [Going deeper with Image Transformers.](http://arxiv.org/abs/2103.17239)
