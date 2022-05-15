---
slug: xcit-part-2
date: 2021-07-25T00:00:00.000Z
title: "[Notes] Understanding XCiT - Part 2"
description: "Local Patch Interaction(LPI) and Class Attention Layer"
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
url: /post/xcit-part-2/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/cat-sweet-kitty-animals-feline-323262/)" >}}

In [Part 1](/post/xcit-part-1), we introduced the XCiT architecture and reviewed the implementation of the Cross-Covariance Attention(XCA) block. In this Part 2, we'll review the implementation of the Local Patch Interaction(LPI) block and the Class Attention layer.

{{< figure src="/post/xcit-part-1/figure-1.png" caption="[from [1]](http://arxiv.org/abs/2106.09681)" >}}

## Local Patch Interaction(LPI)

Because there is no explicit communication between patches(tokens) in XCA, a layer consisting of two depth-wise `3×3` convolutional layers with Batch Normalization with GELU non-linearity is added to enable explicit communication.

Here's the implementation of LPI in [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/xcit.py#L180):

{{< gist ceshine 331714f702c0f6ed064decd21d3896d1 >}}

The module should be very familiar to you if you're versed in traditional convolution networks. Let's first review the initialization of the first convolutional layer:

```python
self.conv1 = torch.nn.Conv2d(
    in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
```

`kernel_size` is 3 by default. `padding` is calculated from `kernel_size` to retain the size of the feature map. The number of groups is set to the number of channels, so there is no interaction between channels (this kind of layers is called “depth-wise convolution layers”).

The second convolutional layer is mostly the same as the first but with a configurable number of output channels (defaults to the number of input channels).

Remember from [Part 1](/post/xcit-part-1) that the output tensor of an XCA block is in `BxNxC` shape. But we're doing 2-D convolution in LPI, so we need to restore it to a proper shape at the start of the `forward` method:

```python
B, N, C = x.shape
x = x.permute(0, 2, 1).reshape(B, C, H, W)
```

The tensor is first permuted to `BxCxN` and then reshaped to `BxCxHxW`. Remember that in vision transformers, an image is split into patches. Each patch is sent through some processing (usually convolutional) layers to be transformed into a vector. It is analogous to looking up the embedding matrix in NLP applications. The vectors from patches are being flattened and treated like tokens in traditional NLP transformers. This step is to reverse the flattening and restore the 2-D structure.

What follows is the usual convolution operations. The first depth-wise convolution is followed by a non-linearity and a batch normalization layer. Then the second convolution is applied.

```python
x = self.conv1(x)
x = self.act(x)
x = self.bn(x)
x = self.conv2(x)
```

At the last step, the tensor is flattened again and permuted back to `BxCxN`.

```python
x = x.reshape(B, C, N).permute(0, 2, 1)
```

As you can see, LPI is just a convolutional block with some additional reshaping and permuting operations to fit into the transformer pipeline.

## Class Attention Layer

This special layer for class attention is introduced in the CaiT architecture[2]. The `CLS` token is added as the input to the first layer in the original vision transformer. This design choice gives the `CLS` token two objectives: (1) guiding the self-attention between patches while (2) summarizing the information useful to the linear classifier. CaiT moves the insertion of the `CLS` token towards the top and freezes the patch embeddings after the insertion.

{{< figure src="cait.png" caption="[from [2]](http://arxiv.org/abs/2106.09681)" >}}

### Class Attention

We first look at [the implementation of the class attention](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/cait.py#L74)(I slightly rearranged the code without affecting the functionality):

{{< gist ceshine 00dfc1307aa30d5f06fec8a85812aece >}}

The query tensor is created only from the first token (`CLS`). The resulting tensor after `unsqueeze` is shaped `Bx1xC`. It is then reshaped to `Bx1xHx(C/H)` and permuted to `BxHx1x(C/H)`. The values in the Q tensor are then divided by $\sqrt{C/H}$ (as in the regular self-attention).

```python
B, N, C = x.shape
q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
q = q * self.scale
```

The key tensor is the standard one. The resulting tensor is shaped `BxHxNx(C/H)`.

```python
k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
```

The followings are standard self-attention calculations:

```python
attn = (q @ k.transpose(-2, -1))
attn = attn.softmax(dim=-1)
attn = self.attn_drop(attn)
```

The first line is the (batch) matrix multiplication between a `BxHx1x(C/H)` tensor and a `BxHx(C/H)xN`, which results in a `BxHx1xN` tensor. The `1xN` part is the column vector containing the attention weight from the `CLS` token to all tokens (`CLS` plus the patches).

Softmax is applied to the last dimension (so the attention weights sum to one), and a dropout layer is applied.

Then the new embedding vector for the `CLS` token is computed according to the attention matrices (they are actually vectors in this case):

```python
x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
x_cls = self.proj(x_cls)
x_cls = self.proj_drop(x_cls)
```

The matrix multiplication in the first line results in a `BxHx1x(C/H)` tensor. After transposing and reshaping(concatenating results from all heads), the tensor's shape becomes `Bx1xC`.

The tensor from the first line goes through another linear transformation and a dropout layer. Then we have the new embedding vector for the `CLS` token!

### Stochastic Depth

Before moving on to the specialized class attention layer, I invite you to review this [DropPath module](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/layers/drop.py#L160) if you're not already familiar with the stochastic depth regularization.

{{< gist ceshine 17068fe5c58951889ded0da720fa7b69 >}}

The dropout mask has a shape of `Bx1x1...`. The exact shape depends on the shape of the input tensor:

```python
shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
random_tensor.floor_()  # binarize
```

The `random_tensor` has range `[keep_prob, 1+keep_prob]`. After binarizing/flooring, the probability of a zero is `1 - keep_prob = drop_prob`, and the probability of a one is `keep_prob`.

The mask (binarized `random_tensor`) is then applied to the input tensor (`.div(keep_prob)` is to keep the mean of the tensor the same after the dropout):

```python
output = x.div(keep_prob) * random_tensor
```

If the mask value is zero, the entire sample will be erased. This should be used on the “main path” (as opposed to the “shortcut path”) in a residual network. The dropped out sample means that the network will only take the shortcut for that sample, which effectively removes one residual layer (hence the name “Stochastic Depth”).

We'll soon see `DropPath` in practice in the next section, so please read ahead even if you're still confused about this concept.

### Class Attention Block

The final specialized class attention layer consists of several [ClassAttentionBlock modules](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/xcit.py#L211):

{{< gist ceshine 7d33799898afce8481f8a94336c621f9 >}}

There are two sets of learnable layer scale parameters (also introduced by CaiT[2]) in the `__init__` methods:

```python
if eta is not None:  # LayerScale Initialization (no layerscale when None)
    self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
    self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
else:
    self.gamma1, self.gamma2 = 1.0, 1.0
```

The `x` input tensor should already have the `CLS` added to its head (with shape `Bx(N+1)xC`). It first goes through one layer of class attention:

```python
x_norm1 = self.norm1(x)
x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
x = x + self.drop_path(self.gamma1 * x_attn)
```

The input tensor is layer normalized. `self.attn(x_norm)` gives us the new `CLS` embedding vectors, which are then concatenated with other tokens `x_norm1[:, 1:]`. The resulting tensor is scaled by `self.gamma1` and sent through a `DropPath` dropout before being added back to the original input tensor `x`.

(If the dropout mask for the particular sample is zero, `x = x + self.drop_path(self.gamma1 * x_attn)` would become `x = x` for that
sample.)

Note that the network can still control the values of patch embedding vectors through `self.gamma1`, so the patch embedding vectors are not strictly frozen.

The tensor `x` goes through another layer normalization. There are two modes implemented: one that normalizes all token vectors; one that only normalizes the `CLS` token vectors (which is the default behavior):

```python
if self.tokens_norm:
    x = self.norm2(x)
else:
    x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
```

The `CLS` token vectors go through a feed-forward network(MLP) and are then scaled again by `self.gamma2`. The new `CLS` token vectors are concatenated with vectors of other tokens, go through a `DropPath`, and are added back to the input tensor `x` (almost the same as in the class attention part, except that the scaling is only applied on `CLS` here):

```python
x_res = x
cls_token = x[:, 0:1]
cls_token = self.gamma2 * self.mlp(cls_token)
x = torch.cat([cls_token, x[:, 1:]], dim=1)
x = x_res + self.drop_path(x)
```

The tensor `x` is returned as the new embedding vectors. We've gone through the entire class attention block!

### Putting it together

Here's how the `ClassAttentionBlock` modules are initialized in [the main XCiT module](https://github.com/rwightman/pytorch-image-models/blob/763329f23f675626e657f012e633fca5ea0985ed/timm/models/xcit.py#L376):

```python
self.cls_attn_blocks = nn.ModuleList([
    ClassAttentionBlock(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        attn_drop=attn_drop_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta, tokens_norm=tokens_norm)
    for _ in range(cls_attn_layers)])
```

And here's the relevant code in the `forward_feature` method:

```python
cls_tokens = self.cls_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, x), dim=1)

for blk in self.cls_attn_blocks:
    x = blk(x)

x = self.norm(x)[:, 0]
return x
```

The initial `CLS` vector is prepended to the tensor `x` before sending it to the `ClassAttentionBlock` modules. The result gets normalized again before returning.

In case it's not clear enough, the result from the `forward_feature` method is then sent through a classifier(a linear layer) to get the logits:

```python
def forward(self, x):
    x = self.forward_features(x)
    x = self.head(x)
    return x
```

## Fin

I hope this two-part series is helpful to you. XCiT is an interesting architecture. The lower memory requirements and its competitive performance (comparing to other STOA models) are all good news to people without access to high-power GPUs (like me). One of the few things I don't like about this paper and the implementation is the hard-coded usage of batch norms in LPI and patch feature extraction layers. It'd be even better if the author can provide model weights pretrained with GroupNorm[3], which is shown to provide more robust performance with tiny batch sizes.

I've also been trying to fine-tune an image classifier using XCiT. The results so far are promising. Be sure to tune the learning rate if you come from the ResNet world. Although XCiT can be understood as “dynamic” convolutions, the usual fine-tuning learning rates for transformers (e.g., `3e-5`) seem to work better than the ones for ResNet.

## References

1. El-Nouby, A., Touvron, H., Caron, M., Bojanowski, P., Douze, M., Joulin, A., … Jegou, H. (2021). [XCiT: Cross-Covariance Image Transformers.](http://arxiv.org/abs/2106.09681)
2. Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., & Jégou, H. (2021). [Going deeper with Image Transformers.](http://arxiv.org/abs/2103.17239)
3. Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., & Houlsby, N. (2019). [Big Transfer (BiT): General Visual Representation Learning.](http://arxiv.org/abs/1912.11370)
