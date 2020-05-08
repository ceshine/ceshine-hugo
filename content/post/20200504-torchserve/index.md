---
slug: torchserve
date: 2020-05-04T00:00:00.000Z
title: "Deploying Efficientnet Model using TorchServe"
description: "A Case Study"
tags:
  - python
  - pytorch
  - deep_learning
  - tips
keywords:
  - python
  - pytorch
  - deep learning
  - tips
url: /post/torchserve/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/TlwP9KxNT18)" >}}

# Introduction

AWS recently released [TorchServe](https://github.com/pytorch/serve), an open-source model serving library for PyTorch. The production-readiness of Tensorflow has long been one of its competitive advantages. TorchServe is PyTorch community's response to that. It is supposed to be the PyTorch counterpart of [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving). So far, it seems to have a very strong start.

[This post from the AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/) and [the documentation of TorchServe](https://github.com/pytorch/serve/blob/master/docs/README.md) should be more than enough to get you started. But for advanced usage, the documentation is a bit chaotic and the example code suggests sometimes conflicting ways to do things.

**This post is not meant to be the tutorials for beginners.** Instead, it uses a case study to show the readers what a slightly more complicated deployment looks like, and saves the readers' time by referencing relevant documents and example code.

In this post,we will deploy an Efficientnet model from the [rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch) repo. The server accepts images as arrays in Numpy binary format and returns the corresponding class probabilities. (The reason for using Numpy binary format is that in this use case the images are already read into memory on the client-side, the network bandwidth is cheap and we don't have strict latency requirements, so re-encoded it into JPEG or PNG format doesn't make sense.)

# Case Study

## Preparing the EfficientNet Model

TorchServe can load models from PyTorch checkpoints (`.state_dict()`) or [exported TorchScript programs](https://pytorch.org/docs/stable/jit.html#creating-torchscript-code). I'd recommend using TorchScript when possible, as it doesn't require you to install extra libraries (e.g., gen-efficientnet-pytorch) and provide a model definition file.

Luckily, `rwightman/gen-efficientnet-pytorch` already provides an easy API to create TorchScript-compatible models. In this case, the model has already been trained and saved via `torch.save(model)`. We need to load it using `torch.load`, create an untrained TorchScript-compatible model, and transfer the weights:

```python
geffnet.config.set_scriptable(True)
model_old = torch.load(
    "cache/b4-checkpoint.pth"
)["model"].cpu()
model, _ = get_model(
    arch="b4", n_classes=6
)
model.load_state_dict(model_old.state_dict())
del model_old
with torch.jit.optimized_execution(True):
    model  = torch.jit.script(model)
model.save("cache/b4.pt")
```

(The above code was inspired by [this script in the gen-efficientnet-pytorch repo](https://github.com/rwightman/gen-efficientnet-pytorch/blob/b16a93756a52e3a7ebd4904358f705f296a1c236/validate.py#L72).)

Please note that the `geffnet.config.set_scriptable(True)` line is essential. Without it the model won't be able to be compiled with TorchScript.

## The Custom Handler

TorchServe comes with [four default handlers](https://github.com/pytorch/serve/blob/1f16a7e5989e046f0e5ce5a9f4c73182f63c4573/docs/default_handlers.md) that define the input and output of the deployed service. We are deploying an image classification model in this example, and the corresponding default handler is `image_classifier`. If you read its [source code](https://github.com/pytorch/serve/blob/1f16a7e598/ts/torch_handler/image_classifier.py), you'll find that it accepts a binary image input, resize, center crop, and normalize it, and returns the top 5 predicted classes. Most of these don't fit our use case, so we'll have to write our own handler. You can refer to [this documentation on how to create non-standard services](https://github.com/pytorch/serve/blob/1f16a7e598/docs/custom_service.md#example-custom-service-file).

### Batch Inference

In this example, we'll have thousands of images per minute to predict, so batch processing is essential. For more information on batch inference with TorchServe, please refer to [this documentation](https://github.com/pytorch/serve/blob/1f16a7e598/docs/batch_inference_with_ts.md).

### Final Result

The code is based on the [resnet_152_batch example](https://github.com/pytorch/serve/blob/master/examples/image_classifier/resnet_152_batch/resnet152_handler.py), with some simplification (e.g., we don't need to handle PyTorch checkpoints). By the way, [the MNIST example](https://github.com/pytorch/serve/blob/master/examples/image_classifier/mnist/mnist_handler.py) used a confusing way to load model and model file, the one in resnet_152_batch makes much more sense (by using the `manifest['model']['serializedFile']` and `manifest['model']['modelFile']` property).

Highlights:

1. Load the model from TorchScript program (Line 30).
1. Load the image from an array in numpy binary format: `input_image = Image.fromarray(np.load(io.BytesIO(image)))` (Line 59).
1. Test Time Augmentation (TTA) in Line 48 to 53, Line 60 to 61, and Line 84 to 87 (horizontal flip).

{{< gist ceshine 15968edd3ac3eaf86b1d2d2775db2760 >}}

This handler cannot handle malformed inputs (as all the example handlers I've seen). If that's inevitable in your use case, you'll probably need to find some way to identify those inputs, ignore that in the `inference` method, and return proper error messages in the `postprocess` method.

## Deployment

### Create the Model Archive

TorchServe requires the user to package all model artifacts into a single model archive file. It's fairly straight-forward in our case. Please refer to [the documentation](https://github.com/pytorch/serve/tree/1f16a7e5989e046f0e5ce5a9f4c73182f63c4573/model-archiver#creating-a-model-archive) if in doubt.

```bash
mkdir model-store
torch-model-archiver --model-name b4 --version 1.0 \
    --serialized-file cache/b4.pt --handler handler.py \
    --export-path model-store
```

### Start the TorchServe service

I use a shell script to start the TorchServe server and register the model:

```bash
torchserve --start --model-store model-store --ts-config config.properties > /dev/null
sleep 3
curl -X DELETE http://localhost:8081/models/b4
curl -X POST "localhost:8081/models?model_name=b4&url=b4.mar&batch_size=4&max_batch_delay=1000&initial_workers=1&synchronous=true"
```

1. `batch_size = 4`: because of the TTA, the effective batch size is actually 8. _I've noticed that the maximum batch size is smaller in TorchServe than directly do the inference in a Python script._ I'm not sure the reason why this is the case.
2. `max_batch_delay=1000`: wait at most 1 second for the batch to be filled. You can adjust this according to your latency requirements.

My `config.properties` file contains:

```text
async_logging=true
vmargs=-Dlog4j.configuration=log4j.properties
```

For now, `log4j.properties` is an exact copy of [the one used by default](https://github.com/pytorch/serve/blob/5982b9f6b436e85eccf9eb2d2ca6468eb4df324d/frontend/server/src/main/resources/log4j.properties).

At this point, your TorchServe service should be up and running.

If you updated your model, create the model archive at the same path, and rerun the shell script will automatically reload the model on the server.

Stop the server by running `torchserve --stop`.

### Client Requests Example

Here's an example of making multiple requests to the server via [asyncio](https://docs.python.org/3/library/asyncio.html):

```python
async def predict_batch(cache):
    buffer = []
    for img in cache["images"]:
        output = io.BytesIO()
        np.save(output, img)
        output.seek(0, 0)
        buffer.append(output)
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    responses = await asyncio.gather(*[
        loop.run_in_executor(
            executor, requests.post, INFERENCE_ENDPOINT, pickled_image
        ) for pickled_image in buffer
    ])
    probs = [res.json() for res in responses]
    return probs
```

# Conclusion

Thanks for reading! I hope this post makes it easier for you to understand and use TorchServe. TorchServe creates an API for your model and does most of the heavy-lifting involved in handling HTTP requests. It shows great promise in the production environment support of PyTorch.

If you have any suggestions, please feel free to leave a comment.
