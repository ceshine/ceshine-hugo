---
slug: fastcore-patch-to-case-study
date: 2021-02-19T00:00:00.000Z
title: "A Case Study of fastcore @patch_to"
description: "Trying out SnapMix with minimal changes to the codebase"
tags:
  - python
  - pytorch
  - tip
keywords:
  - python
  - pytorch
  - tip
url: /post/fastcore-patch-to-case-study/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/m9V4GFBJ-4g)" >}}

## Motivation

I recently came across this new image data augmentation technique called [SnapMix](https://arxiv.org/abs/2012.04846). It looks like a very sensible improvement over [CutMix](https://arxiv.org/abs/1905.04899), so I was eager to give it a try.

The SnapMix author provides a PyTorch implementation. I made some adjustments to improve the numeric stability and converted it to [a callback in PyTorch Lightning](https://github.com/veritable-tech/pytorch-lightning-spells/blob/7ea3de2beeafb98bf83fb0b643ac575a59707305/pytorch_lightning_spells/callbacks.py#L42). I encountered one major obstacle during the process — SnapMix uses Class Activation Mapping(CAM) to calculate an augmented example's label weights. It requires access to the final linear classifier's weight and the model activations before the pooling operation. Some PyTorch pre-trained CV models do implement methods to access these two things, but the namings are inconsistent. We need a unified API to do this.

One way to create a unified API is to subclass every pre-trained model and implement `get_fc` and `extract_features` methods. However, this requires switching existing imports to the new subclasses. As I want to test SnapMix to see if it works quickly, I want to change the existing codebase as little as possible. Here's where `@patch_to` from _fastcore_ came in.

[The _fastcore_ library](https://fastcore.fast.ai/) is a spin-off of the _fast.ai_ library. It provides some useful power-ups to the Python standard library, and `@patch_to` is one of them.

## The Solution

The following code block shows how I patch the EfficientNet class from `getn_efficientnet` to make it compatible with SnapMix:

```python
import geffnet
from fastcore.basics import patch_to


@patch_to(geffnet.gen_efficientnet.GenEfficientNet)
def extract_features(self, input_tensor):
    return self.features(input_tensor)


@patch_to(geffnet.gen_efficientnet.GenEfficientNet)
def get_fc(self):
    return self.classifier
```

Pretty neat, isn't it? Another advantage of this approach is that it supports patching multiple classes in one call. For example, if we have another PyTorch class that also stores its final classifier in `self.classifier`, we can pass it along with the existing class as a tuple to the `@patch_to` decorator.

## More Details

[The documentation](https://fastcore.fast.ai/basics.html#patch_to) and [the source code](https://github.com/fastai/fastcore/blob/875988a7ed359a3eb16fd2166bf8fb42b190881c/fastcore/basics.py#L762) of the fastcore library can be a bit confusing. Therefore, I create a small demo script to showcase the ability of the `@patch_to` decorator:

{{< gist ceshine 318476922b8ed42aa7aaa7e0fac70c98 >}}

## Conclusion

That's it. Thanks for reading this short post. Let me know if you can think of a better way to patch the PyTorch model. I'd also love to know your experience using the `fastcore` library. Leave a comment if you have any questions. I'd usually answer within a week (DM me on Twitter if that doesn't happen. Sometimes I can miss the notification.).

More Details

The documentation and the source code of the fastcore library can be a bit confusing. Therefore, I create a small demo script to showcase the ability of the @patch_to decorator:
