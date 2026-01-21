---
slug: colpali-preprocessing
date: 2026-01-22T00:00:00.000Z
title: "[Notes] ColPali's Image Preprocessing Pipeline"
description: ""
tags:
  - python
  - nlp
  - pytorch
  - transformers
  - cv
keywords:
  - Python
  - NLP
  - Computer Vision
  - RAG
  - ColPali
  - Qdrant
url: /post/colpali-preprocessing/
---

## A Brief Review of Model Implementations

The ColPali class is essentially a PaliGemma model with a custom linear projection layer (the `custom_text_proj` attribute).

There are four variants of ColPali at the time of writing:

1. [vidore/colpali](https://huggingface.co/vidore/colpali)
2. [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1)
3. [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)
4. [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3)

According to the [adapter_config.json](https://huggingface.co/vidore/colpali-v1.3/blob/main/adapter_config.json), they were all fine-tuned with LoRA on the same base model — [vidore/colpaligemma-3b-pt-448-base](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base).

Judging from the `target_modules` string, the fine-tuned components include all linear projection layers in the language-model portion of PaliGemma, as well as the final custom linear projection layer.

```json
{
  "target_modules": (.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)
}
```

The [vidore/colpaligemma-3b-pt-448-base](https://huggingface.co/vidore/colpaligemma-3b-pt-448-base) model seems to be just the PaliGemma-3B model weights ported into the format used by the ColPali class.

{{< figure src="ColPali-model-class.png" caption="Analyze [the ColPali Class](https://github.com/illuin-tech/colpali/blob/51ccfe76ac2c124aa463f244390440cd313985bd/colpali_engine/models/paligemma/colpali/modeling_colpali.py#L12)" >}}

A quick glance at the newer [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0/blob/main/adapter_config.json) models shows a similar structure: a ported base model ([vidore/colqwen2.5-base](https://huggingface.co/vidore/colqwen2.5-base)) with a linear projection layer at the top, and LoRA-fine-tuned variants.
