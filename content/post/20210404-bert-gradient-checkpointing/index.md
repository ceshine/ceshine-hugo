---
slug: bert-gradient-checkpoint
date: 2021-04-04T00:00:00.000Z
title: "[Notes] Gradient Checkpointing with BERT"
description: "A brief analysis of huggingface's implementation"
tags:
  - pytorch
  - nlp
  - notes
keywords:
  - pytorch
  - nlp
  - codethrough
url: /post/bert-gradient-checkpoint/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/7Hu4iWksw2k)" >}}

## Overview

Gradient checkpointing is a technique that reduces the memory footprint during model training (From `O(n)` to `O(sqrt(n))` in the OpenAI example, `n` being the number of layers). The price is some computing overhead (multiple forward-pass on the same input). [This post by Yaroslav Bulatov of OpenAI](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) explains the mechanism behind it very well.

{{< figure src="memory-usage-decomposition.png" caption="[Source: Gugger's slides](https://www.slideshare.net/SylvainGugger/fine-tuning-large-lms-243430468)" >}}

In many cases, what consumes the most memory is not the model itself but the intermediate activations and gradients of them, as [this set of slides by Sylvain Gugger](https://www.slideshare.net/SylvainGugger/fine-tuning-large-lms-243430468) shows. Gradient checkpointing replaces the intermediate activations with checkpoints (the model is split into chunks by checkpoints) and recreates the activations between checkpoints by running another forward-pass in this chunk. Every activation is computed at most twice (once in the last chunk, twice in others). We only need to store the checkpoints (also a set of activations) and the activations of the active chunk in the memory during the backward-pass. If we're using a model with `n` layers of equal size and we put a checkpoint every 10 layers (9 checkpoints, at layer 10, 20, ..., 90.), memory consumption from activations and gradients of them is `(9+10)kn` comparing to `100kn` without checkpointing (`k` is a constant).

## BERT Implementation

PyTorch now natively [supports gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html). Please refer to [this post by Qingyang Wu](https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html) for an analysis of the implementation.

The [Transformers](https://github.com/huggingface/transformers) library from huggingface supports gradient checkpointing in some of the models. The following is [how it is done for BERT](https://github.com/huggingface/transformers/blob/cd56f3fe7eae4a53a9880e3f5e8f91877a78271c/src/transformers/models/bert/modeling_bert.py#L544):

```python
if getattr(self.config, "gradient_checkpointing", False) and self.training:

    if use_cache:
        logger.warn(
            "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
            "`use_cache=False`..."
        )
        use_cache = False

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs, past_key_value, output_attentions)

        return custom_forward

    layer_outputs = torch.utils.checkpoint.checkpoint(
        create_custom_forward(layer_module),
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    )

else:

    layer_outputs = layer_module(
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        output_attentions,
    )
```

The only difference between `gradient_checkpointing=True` and `gradient_checkpointing=False` is that the former use [a closure](https://www.programiz.com/python-programming/closure) to wrap calls to the `BertLayer` module. This is becayse the `checkpoint` function only supports tensors or None as function call arguments, so we use closures to pass other types of values into the call.

One of the limitations of this approach is that it's not generalizable to other kinds of transformer models. We need to add checkpointing mechanism in every model implementation manually. It explains why gradient checkpointing is not supported in some newer models.

The other limitation is that this implementation only allows us to put a checkpoint before each `BertLayer` module. We don't have an option to set checkpoints (e.g., every other two layers). But given that the transformer layers are already relatively large, this limitation should not be too big of a hurdle in most cases.

(There was [an attempt at a general/model-agnostic gradient checkpointing implementation](https://github.com/huggingface/transformers/pull/5415), but it has gone stale after a few months.)
