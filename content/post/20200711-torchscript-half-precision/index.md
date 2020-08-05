---
slug: torchscript-half-precision
date: 2020-07-11T00:00:00.000Z
title: "[Tip] TorchScript Supports Half Precision"
description: "Speeding up inference for models trained with mixed precision"
tags:
  - deep-learning
  - pytorch
  - tips
keywords:
  - pytorch
  - torchscript
url: /post/torchscript-half-precision/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/wdc9ZAiwBB4)" >}}

This is a short post describing how to use half precision in [TorchScript](https://PyTorch.org/docs/stable/jit.html). This can speed up models that were trained using mixed precision in PyTorch (using [Apex Amps](https://github.com/NVIDIA/apex)), and also some of the model trained using full precision (with some potential degradation of accuracy).

> TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency. [source](https://PyTorch.org/docs/stable/jit.html#)

> This repository (NVIDIA/apex) holds NVIDIA-maintained utilities to streamline mixed precision and distributed training in PyTorch. Some of the code here will be included in upstream PyTorch eventually. [source](https://github.com/NVIDIA/apex)

# Overview

One thing that I managed to forget is that **PyTorch itself already supports half precision computation**. I wanted to speed up inference for my TorchScript model using half precision, and I spent quite some time digging around before it came to me. It doesn't need Apex Amp to do that. What Amp does for you is patching some of the PyTorch operation so only they run in half precision (O1 mode), or keep master weights in full precision and run all other operations in half (O2 mode, see the diagram below). It also handles the scaling of gradients for you. These are all essential in mixed precision training.

{{< figure src="diagram.png" caption="Mixed Precision Training [[source]](https://medium.com/datadriveninvestor/mixed-precision-training-for-deep-neural-networks-3751f2c88883)" >}}

But when you finished training and wants to deploy the model, almost all the features provided by Apex Amp are not useful for inference. So you don't really need the Amp module anymore. Besides, you can not use Apex Amp in TorchScript, so you don't really have a choice. Simply convert the model weights to half precision would do.

# Examples

Below I give two examples of converting a model weights and then export to TorchScript.

[BiT-M-R101x1](https://github.com/google-research/big_transfer) Model:

```python
from bit_models import KNOW_MODELS

model = KNOWN_MODELS["BiT-M-R101x1"](head_size=100)
model.eval()
model.half()
model.load_state_dict(torch.load(
    "../cache/BiT-M-R101x1.pth"
)["model_states"])
with torch.jit.optimized_execution(True):
    model  = torch.jit.script(model)
model.save("../cache/BiT-M-R101x1.pt")
```

[EfficientNet-B4](https://github.com/rwightman/gen-efficientnet-pytorch) Model:

```python
import geffnet
geffnet.config.set_scriptable(True)
model = geffnet.tf_efficientnet_b4_ns(pretrained=False, as_sequential=True)
model.load_state_dict(torch.load(
    "../cache/b4.pth"
)["model"].cpu().state_dict())
model.eval()
with torch.jit.optimized_execution(True):
    model  = torch.jit.script(model)
model.save("../cache/b4.pt")
```

You'll need to convert the input tensors. I also convert the logits back to full precision before the Softmax as it's a recommended practice. This is what I do in the evaluation script:

TL;DR version:

```python
if half:
    input_tensor = input_tensor.half()
probs = F.softmax(model(input_tensor).float(), -1)
```

Full version:

```python
def collect_predictions(model, loader, half: bool):
    model.eval()
    outputs, y_global = [], []
    with torch.no_grad():
        for input_tensor, y_local in tqdm(loader, ncols=100):
            batch_size = input_tensor.size(0)
            aug_size = input_tensor.size(1)
            if half:
                input_tensor = input_tensor.half()
            input_tensor = input_tensor.view(
                -1, *input_tensor.size()[2:]).to("cuda:0")
            tmp = F.softmax(model(input_tensor).float(), -1).cpu()
            probs = torch.mean(
                tmp.view(batch_size, aug_size, -1), dim=1).clamp(1e-5, 1-1e-5)
            outputs.append(probs)
            y_global.append(y_local.cpu())
        outputs = torch.cat(outputs, dim=0)
        y_global = torch.cat(y_global, dim=0)
    return outputs, y_global
```

## Simple Benchmarks

(The model were evaluated on a private image classification dataset. The model were trained in Apex O2 mode.)

| Mode               | Time (seconds) | Loss         |
| ------------------ | -------------- | ------------ |
| TorchScript (fp16) | **51**         | **0.379376** |
| Apex (O1)          | 72             | 0.379236     |
| Apex (O2)          | **49**         | **0.379376** |
| Apex (O3)          | 72             | 0.379346     |
| PyTorch FP32       | 86             | 0.379291     |

Remarks:

- Apex (O2) and TorchScript (fp16) got exactly the same loss, as they should. The feed-forward computation are exactly the same in these two modes.
- Apex (O3) is surprisingly slow. Not sure why.

# Bonus: TRTorch

[TRTorch](https://github.com/NVIDIA/TRTorch) is a new tool developed by NVIDIA and converts a standard TorchScript program into an module targeting a TensorRT engine. With this, we will not need to export the PyTorch model to ONNX format to run model on [TensorRT](https://developer.nvidia.com/tensorrt) and speed up inference.

However, TRTorch still does not support at lot of operations. Both the BiT-M-R101x1 model and the EfficientNet-B4 model failed to be compiled by TRTorch, making it's not very useful for now. But I really like this approach, and wish this projects gain more momentum soon.
