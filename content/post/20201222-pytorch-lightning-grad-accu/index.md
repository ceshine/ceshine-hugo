---
slug: pytorch-lightning-grad-accu
date: 2020-12-22T00:00:00.000Z
title: "[PyTorch Lightning] Log Training Losses when Accumulating Gradients"
description: "The global step is not what you think it is"
tags:
  - pytorch
  - pytorch-lightning
keywords:
  - tip
  - pytorch
  - pytorch lightning
url: /post/pytorch-lightning-grad-accu/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/music-recording-vinyl-retro-disk-5705801/)" >}}

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) reached 1.0.0 on October 2020. I wasn't fully satisfied with the flexibility of its API, so I continued to use my own [pytorch-helper-bot](https://github.com/ceshine/pytorch-helper-bot/). This has changed since the 1.0.0 release. Now I use PyTorch Lightning to develop training code that supports both single and multi-GPU trainings.

However, one thing that bugged me is that **the logging doesn't work correctly when I set the number of gradient accumulation batches larger than one**. The steps recorded in the training loop is still the raw step number, but those recorded in the validation is divided by the number of gradient accumulation batches. The training loop will be flooded with warnings of inconsistent steps being recorded. And it'll be harder for you to compare the training and validation losses without the same step scale.

The support and documentation for gradient accumulation does not seem sufficient at this moment. I digged around the PyTorch Lightning source code, did some experiments, and found some workarounds for this issue.

## The Wrong Way

Let's first see a naive (and mostly wrong) way to log the training losses:

```python
def training_step(self, batch, batch_idx: int) -> dict:
    inputs, targets = batch
    logits = self.forward(inputs)
    loss = F.cross_entropy(logits, targets)
    self.log('train_loss', loss)
    return {'loss': loss}
```

This only works when you have `accumulate_grad_batches=1` in the trainer. The steps associated with `train_loss` will be `n` times larger than the global step if you set `accumulate_grad_batches` to `n`.

## Attempt #1

### Context: the global step

One thing that confused me was the definition of step number (found at `self.global_step`) by PyTorch Lightning. In PyTorch Lightning, a step is counted when the `optimizer.step` method is called, not when `loss.backward` is called. So if you have `accumulate_grad_batches=2` and have trained 10 batches, the counted steps is 5, not 10.

What we want is to match the step number of a training loss with the global step variable.

### Implementation

Inspired by the implementation of the official callback [LearningRateMonitor](https://github.com/PyTorchLightning/pytorch-lightning/blob/43f73fdfdbd0d980031a9acc867c0cc362448a63/pytorch_lightning/callbacks/lr_monitor.py#L31), we can try to explicitly set the step by directly calling the `log_metrics` method:

```python
def training_step(self, batch, batch_idx: int) -> dict:
    inputs, targets = batch
    logits = self.forward(inputs)
    loss = F.cross_entropy(logits, targets)
    # the new line
    self.logger.log_metrics({"train_loss": loss}, step=trainer.global_step)
    return {'loss': loss}
```

The step number is correct now, but we now have too many data points! The training loss of every step is recorded and in most case it's not what we want.

## Attempt #2

Again inspired by [LearningRateMonitor](https://github.com/PyTorchLightning/pytorch-lightning/blob/43f73fdfdbd0d980031a9acc867c0cc362448a63/pytorch_lightning/callbacks/lr_monitor.py#L31), we can use the `log_every_n_steps` attribute in the trainer to reduce the number of data points:

```python
def training_step(self, batch, batch_idx: int) -> dict:
    inputs, targets = batch
    logits = self.forward(inputs)
    loss = F.cross_entropy(logits, targets)
    # The new lines
    should_log = (
        (self.global_step + 1) % self.log_every_n_steps == 0
    )
    if should_log:
      self.logger.log_metrics({"train_loss": loss}, step=trainer.global_step)
    return {'loss': loss}
```

Now the number of data points are down by a lot. We'll be able to see another problem in the visualized data — we have multiple data points on the same step. This is because for each `global_step`, `training_step` will called `n` times, with `n` being the number of batches to accumulate.

## Attempt #3 (Good for a single GPU)

We're almost there. We can use `batch_idx` to help us only record one data point per one optimizer step(a.k.a. `global_step`):

```python
def training_step(self, batch, batch_idx: int) -> dict:
    inputs, targets = batch
    logits = self.forward(inputs)
    loss = F.cross_entropy(logits, targets)
    # The new line
    if batch_idx % self.trainer.accumulate_grad_batches == 0:
      should_log = (
          (self.global_step + 1) % self.log_every_n_steps == 0
      )
      if should_log:
        self.logger.log_metrics({"train_loss": loss}, step=trainer.global_step)
    return {'loss': loss}
```

Now the logging will work properly when you are training on a single GPU

## Attempt #4 (EMA)

So far we're logging only samples of the training losses. The sampled losses have higher variance and less reliability. A better way to do this is to log the smoothed version of the training losses. For example, we can use exponential moving averages:

```python
class EMATracker:
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        if self._value is None:
            self._value = new_value
        else:
            self._value = (
                new_value * self.alpha +
                self._value * (1-self.alpha)
            )

    @property
    def value(self):
        return self._value

class ExampleModule(pytorch_lightning.LightningModule):
    def __init__(self, ...):
        ...
        self.train_loss_tracker = EMATracker(alpha=0.02)

    def training_step(self, batch, batch_idx: int) -> dict:
      inputs, targets = batch
      logits = self.forward(inputs)
      loss = F.cross_entropy(logits, targets)
      # A new line
      self.train_loss_tracker.update(loss)
      if batch_idx % self.trainer.accumulate_grad_batches == 0:
        should_log = (
            (self.global_step + 1) % self.log_every_n_steps == 0
        )
        if should_log:
          # A new line
          self.logger.log_metrics({
            "train_loss": self.train_loss_tracker.value
          }, step=trainer.global_step)
      return {'loss': loss}
```

The training losses recorded will now take all losses into account and be much smoother.

## Attempt #5 (Good for multiple GPUs)

The above code will create inaccurate results when training on multiple GPUs. We'll need to aggregate the losses from all GPUs in the `training_step_end` method before logging:

```python
def _should_log(self, flag):
    if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
        if isinstance(flag, list):
            return flag[0]
        return flag
    return False

def training_step_end(self, outputs):
    loss = outputs["loss"].mean()
    self.train_loss_tracker.update(loss)
    if self._should_log(outputs["log"]):
        self.logger.log_metrics({
            "train_loss": self.train_loss_tracker.value
        }, step=self.global_step)
    return loss

def training_step(self, batch, batch_idx):
    inputs, targets = batch
    logits = self.forward(inputs)
    loss = F.cross_entropy(logits, targets)
    return {"loss": loss, "log": batch_idx % self.trainer.accumulate_grad_batches == 0}
```

(Author note: this part has not been tested on a multi-GPU environment yet. Will update once it has been tested.)

## A More General Solution (WIP)

As you can see, there's a lot of coding involved to make the logging work. We'll have to create a new EMATracker for a new metric we want to track, and add the needed code in the `training_step` and `training_step_end` methods.

Using a callback to do this for us would be more scalable solution. We can create a new callback for each new metric and plug it to the trainer. Unfortunately, the callback hook `on_train_batch_end` currently does not get passed the batch outputs at every step, so it's not possible to do it using the internal callback API.

There's already [a pull request](https://github.com/PyTorchLightning/pytorch-lightning/pull/4369) addressing this issue. We'll come back to this section once the pull request has been merged.
