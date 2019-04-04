---
slug: custom-image-augmentation-with-keras
date: 2019-04-04T04:19:04.027Z
title: "Custom Image Augmentation with Keras"
description: "Solving CIFAR-10 with Albumentations and TPU on Google Colab"
images:
  - featuredImage.jpeg
  - 1*eQF6vJx8Ke_WbfTsHvdZJw.png
  - 1*S_IXy4CSKSCTLHj_X9lGWQ.png
tags:
  - machine_learning
  - image_classification
  - keras
  - tensorflow
  - tpu
keywords:
  - machine-learning
  - deep-learning
  - image-classification
  - keras
  - tensorflow
url: /post/custom-image-augmentation-with-keras/
---

{{< figure src="featuredImage.jpeg" caption="Photo by [Josh Gordon](https://unsplash.com/photos/fw9cbA1WTi0?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)" >}}

The new Tensorflow 2.0 is going [to standardize on Keras as its High-level API](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a). The existing Keras API will mostly remain the same, while Tensorflow features like eager execution, distributed training and other deeper Tensorflow integration will be added or improved. I think it’s a good time to revisit Keras as someone who had switched to use PyTorch most of the time.

I wrote [an article benchmarking the TPU on Google Colab with the Fashion-MNIST dataset](https://medium.com/the-artificial-impostor/keras-for-tpus-on-google-colaboratory-free-7c00961fed69) when Colab just started to provide TPU runtime. This time I’ll use a larger dataset ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)) and an external image augmentation library `[albumentation](https://github.com/albu/albumentations)s`.

It turns out that implementing a custom image augmentation pipeline is fairly easy in the newer Keras. We could give up some flexibility in PyTorch in exchange of the speed up brought by TPU, which is not yet supported by PyTorch yet.

# Source Code

* [GPU version](https://colab.research.google.com/drive/1zTYNJ3xtPeNsa5ARBw4Ufj8crPZJx-cp) (with a Tensorboard interface powered by `ngrok`)

* [TPU version](https://colab.research.google.com/drive/1hFFzWabe5sI3vO92AIqPi4P-0DoYDN0U)

The notebooks are largely based on the work by [Jannik Zürn](https://medium.com/@jannik.zuern) described in this post:
**[Using a TPU in Google Colab](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)**.

I updated the model architecture from [the official Keras example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) and modified some of the data preparation code.

# Custom Augmentation using the Sequence API

From the Keras documentation:

> [`Sequence`](https://keras.io/utils/) are a safer way to do multiprocessing. This structure guarantees that the network will only train once on each sample per epoch which is not the case with generators.

Most Keras tutorials use the [ImageDataGenerator](https://keras.io/preprocessing/image/) class to generate batch and do image augmentation. But it doesn’t leave much room for customization (unless you spend some time reading the source code and extend the class) and the augmentation toolbox might not be comprehensive or [fast enough](https://github.com/albu/albumentations#benchmarking-results) for you.

## Class Definition

Fortunately, there’s a `Sequence` class (`keras.utils.Sequence`) in Keras that is very similar to [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class in PyTorch (although Keras doesn’t seem to have its own [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)). We can construct our own data augmentation pipeline like this:

{{< gist ceshine 3d77c5dd6beb8e30743436e74c612767 >}}

Note the one major difference between `Sequence` and `Dataset` is that `Sequence` returns an entire batch, while `Dataset` returns a single entry.

In this example, the data has already been read in as `numpy` arrays. For larger datasets, you can store paths to the image files and labels in the file system in the class constructor, and read the images dynamically in the `__getitem__` method via one of the two methods:

* OpenCV:`cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_RGB2BGR)`

* PIL: `np.array(Image.open(filepath))`

Reference: [An example pipeline that uses torchvision](https://github.com/albu/albumentations/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb).

## Albumentations

Now we use albumentations to define a set of augmentations to be applied randomly to training set and a (deterministic) set for the test and validation sets:

{{< gist ceshine ff2a24c4d34e0fd22e8bf5fc8e5b793d >}}

{{< figure src="1*eQF6vJx8Ke_WbfTsHvdZJw.png" caption="Augmented Samples" >}}

**[`ToFloat(max_value=255)`](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToFloat)** transforms the array from [0, 255] range to [0, 1] range. If you are tuning a pretrained model, you’ll want to use **[`Normalize`](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Normalize)** to set `mean` and `std`.

## Training and Validating

Just pass the sequence instances to the `fit_generator` method of an initialized model, Keras will do the rest for you:

{{< gist ceshine e156c7a50cc81b8696fbcaf134832dd5 >}}

By default Keras will shuffle the batches after one epoch. You can also choose to shuffle the entire dataset instead by implementing a `on_epoch_end` method in your `Sequence` class. You can also use this method to do other dynamic transformations to the dataset between epochs (as long as the `__len__` stay the same, I assume).

That’s it. You now have a working customized image augmentation pipeline.

# TPU on Google Colab

{{< figure src="1*S_IXy4CSKSCTLHj_X9lGWQ.png" caption="Model used: [Resnet101 v2 in the official example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)" >}}

Notes to the table:

1. The sets of augmentations used by GPU and TPU notebook are slightly different. The GPU one includes a [`CLAHE`](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CLAHE) op while the TPU one does not. This is due to an oversight on my part.

1. The GTX 1080 Ti results are taken from [the official example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py).

The batch size used by Colab TPU is increased to utilize the significantly larger memory size (64GB) and TPU cores (8). Each core will received 1/8 of the batch.

## Converting Keras Models to use TPU

Like before, one single command is enough to do the conversion:

{{< gist ceshine 1100e528c1fa5d5806095a98b4c96292 >}}

But because the training pipeline is more complicated than the Fashion-MNIST one, I encountered a few obstacles, and had to find ways to circumvent them:

1. The runtime randomly hangs or crashes when I turn on `multiprocessing=True `in `fit_generator` method, despite the fact that `Sequence `instances should support multiprocessing.

1. The TPU backend crashes when Keras has finished first epoch of training and starts to run validation.

1. No good way to schedule training rate. The TPU model only supports `tf.train` optimizers, but on the other hand the Keras learning rate schedulers only support Keras optimizers.

1. The model gets compiled four times (two when training, two when validating) at the beginning of `fit_generator `call, and the compile time is fairly long and unstable (high variance between runs).

The corresponding solutions:

1. Use `multiprocessing=False`. This one is obvious.

1. Run a “warmup” round of one epoch without validation data seems to solve the problem.

1. The Tensorflow 2.0 version of Keras optimizer seems to work with TPU models. But as we’re using the pre-installed Tensorflow 1.13.1 on Colab, one hacky solution is to **sync the TPU model to CPU and recompile the model using an optimizer with a lower learning rate**. This is not ideal, of course. We’d waste 5 ~ 20 minutes syncing and recompiling the model.

1. This one unfortunately I couldn’t find good way to avoid it. The reason why the model get compiled four times is because the last batch has a different size from the previous ones. We could reduce the number to three if we just drop the last batch in training (I couldn’t find a way to do that properly in Keras). Or reduce the number to two if we pick a batch size that is a divisor to the size of the dataset, which is not always possible or efficient. You could just throw away some data to make things easier if your dataset is large enough.

# Summary

The TPU (TPUv2 on Google Colab) greatly reduces the time needed to train an adequate model, albeit its overhead. But get ready to deal with unexpected problems since everything is really still experimental. It was really frustrating for me when the TPU backend kept crashing for no obvious reason.

The set of augmentations used here is relatively mild. There are a lot more options in the `albumentations` library (e.g. [Cutout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Cutout)) for you to try.

If you found TPU working great for you, [the current pricing of TPU ](https://cloud.google.com/tpu/docs/pricing)is quite affordable for a few hours of training (Regular $4.5 per hour and preemptible **$1.35** per hour). (I’m not affiliated with Google.)

In the future I’ll probably try to update the notebooks to Tensorflow 2.0 alpha or the later RC and report back anything interesting.