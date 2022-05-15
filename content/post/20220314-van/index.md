---
slug: van-notes
date: 2022-03-14T00:00:00.000Z
title: "[Notes] Understanding Visual Attention Network"
description: "Decompose large kernel convolutions to get attention weights efficiently"
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
url: /post/van-notes/
---

{{< figure src="featuredImage.jpg" caption="[credit](https://unsplash.com/photos/VULVydw3nV0)" >}}

## Introduction

At the start of 2022, we have a new pure convolution architecture ([ConvNext](/post/convnext-notes/))[1] that challenges the transformer architectures as a generic vision backbone. The new Visual Attention Network (VAN)[2] is yet another pure and simplistic convolution architecture that its creators claim to have achieved SOTA results with fewer parameters.

<div style="max-width: 600px; margin-left: auto; margin-right: auto;">{{< figure src="fig-1.png" caption="Source: [2]" >}}</div>

What ConvNext tries to achieve is modernizing a standard ConvNet (ResNet) without introducing any attention-based modules. VAN still has attention-based modules, but the attention weights are obtained from a large kernel convolution instead of a self-attention block. To overcome the high computation costs brought by a large kernel convolution, it is decomposed into three components: a spatial local convolution (depth-wise convolution), a spatial long-range convolution (depth-wise dilation convolution), and a channel convolution (1x1 point-wise convolution).

<div style="max-width: 600px; margin-left: auto; margin-right: auto;">{{< figure src="fig-2.png" caption="Source: [2]" >}}</div>

The authors propose an attention module based on this decomposition called “Large Kernel Attention.” This attention module plays the central role in their Visual Attention Network, where LKA is surrounded by two 1x1 convolutions and a GELU activation. There are also two residual connections in each of the _L_ groups in each stage that are not shown in the figure. We'll learn more about the implementation details in the next section.

<div style="max-width: 600px; margin-left: auto; margin-right: auto;">{{< figure src="fig-3.png" caption="Source: [2]" >}}</div>

## Code Analysis

The authors have open-sourced [a VAN implementation for image classification on GitHub](https://github.com/Visual-Attention-Network/VAN-Classification). Generally speaking, the code is very readable. There are some parts that seem to be leftovers from a ViT-like model the author created this implementation from that contribute nothing to the data flow. I'll skip those parts below to avoid confusion. The `_init_weights` parts are also skipped for brevity.

### Overall data flow

Let's take a top-down view of the data flow to have a general idea of what kinds of components are involved. (The `__init__` method is skipped for now.)

```python
class VAN(nn.Module):

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

The `patch_embdd` module is the downsampling block, and the `block` module corresponds to the part that is enclosed in dotted lines in Fig 3. Each `block` contains _L_ groups of modules.

The shape of `x` is `(B, C, H, W)` for `patch_embed` and `block`. The tensor is flattened and transposed into the shape of `(B, H*W, C)` and a layer normalization is **applied on the last (channel) dimension**. If it has not reached the last stage, the normalized tensor is then reshaped back into `(B, C, H, W)`. (This reshaping and transposing/permuting of the tensor might negatively affect the efficiency. The official implementation of ConvNext created a custom LayerNorm to avoid these operations altogether.)

### Downsampling Layer

I believe the name `OverlapPatchEmbed` is from the nomenclature of ViT-based models. It's just a regular downsampling layer with a convolution with a stride of 4 (the first layer) or 2 (the rest) and a batch normalization afterward. Note that, unlike ConvNext, the “patches” overlap with each other, hence the `Overlap` in the module name.

```python
class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W
```

### Main Block

[DropPath](https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L135) is taken from the [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/) library (a.k.a. _timm_). It randomly drops the entire sample from the tensor. Combined with a residual connection, this would mean nothing will be added to that sample, in effect skipping the entire layer. That's why this technique is also called “stochastic depth.”

The two residual connections are marked by the two batch normalizations. The first residual block starts right before the first batch normalization and ends before the second. The second residual block comes right after the first block and ends after the MLP (CFF) module. Two scaling vectors are applied to the residuals channel-wise before they are added to the original values.

```python
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x
```

### Convolution Feed-Forward

This looks very similar to the feed-forward layer in transformers, with an additional 3x3 depth-wise convolution layer after the first `1x1 point-wise convolution layer.

```python
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        # pointwise
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
```

### Spatial Attention Layer

This `SpatialAttention` module implements the part after the first batch normalization to the second (1x1, GELU, LKA, 1x1). Note that there is one more residual connection inside it. This connection does not seem necessary to me. If the inner connection has a zero residual value, then the batch-normalized value will be added back to the original value from the outer connection, which does not make much sense to me.

```python
class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # Depthwise dilation convolution
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        # pointwise convolution
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
```

### VAN initialization

Finally, let's circle back to the initialization of the VAN module. The only thing that might be a bit confusing is the `dpr` list. The idea is to increase the `DropPath` probability as we progress into later stages. We don't want to skip earlier stages because they process local information (e.g., edges) that are more essential than the global information processed in later stages.

```python
lass VAN(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        flag=False,
    ):
        super().__init__()
        if flag is False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )
```

## Conclusion

Visual Attention Network is elegantly designed and has very good performance and efficiency on paper. However, I wish there is some throughput benchmarking as in the ConvNext paper. The heavy use of depth-wise convolutions can significantly drag down the training speed. I've already observed that the tiny version of VAN is not much faster than the small version in my preliminary experiments, probably because of the bottleneck in memory bandwidth instead of computation.

Nonetheless, it's still very impressive that such simple architecture can achieve this level of accuracy. I'm looking forward to more research in this direction.

<div style="max-width: 600px; margin-left: auto; margin-right: auto;">{{< figure src="table-1.png" caption="Source: [2]" >}}</div>

## References

1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). [A ConvNet for the 2020s](http://arxiv.org/abs/2201.03545).
2. Guo, M.-H., Lu, C.-Z., Liu, Z.-N., Cheng, M.-M., & Hu, S.-M. (2022). [Visual Attention Network](https://arxiv.org/abs/2202.09741 ).
