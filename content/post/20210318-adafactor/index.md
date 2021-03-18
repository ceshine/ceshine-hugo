---
slug: adafactor
date: 2021-03-18T00:00:00.000Z
title: "[Paper] Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
description: "Essential for fine-tuning T5 v1.1 and mT5 models"
tags:
  - pytorch
  - nlp
  - paper
keywords:
  - pytorch
  - nlp
  - paper
  - research
  - codethrough
url: /post/adafactor/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/POOKR7hxtuU)" >}}

## Motivation

The [Adafactor optimizer](https://arxiv.org/abs/1804.04235), in my experience, can provide much better convergence than fine-tuning the [T5 v1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511) and [mT5](https://arxiv.org/abs/2010.11934)[1] pre-trained models. However, I encountered problems when using a custom learning rate scheduler with the Adafactor implementation from the [huggingface/transformer](https://github.com/huggingface/transformers) library. I combed through the paper and the source code to find and fix the cause of the problem, which turned into a [tiny contribution](https://github.com/huggingface/transformers/pull/9751) to the library.

To further squeeze value from the time I've invested, I wrote this post to introduce the key ideas of the Adafactor optimizer and analyze the corresponding chunk of code in the huggingface/transformer implementation (which was taken from the [fairseq library](https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py)). Working examples as Kaggle notebooks are also provided: [T5 v1.1](https://www.kaggle.com/ceshine/preprocess-and-finetune-t5-1-1-full/) and [mT5](https://www.kaggle.com/ceshine/preprocess-and-finetune-mt5).

(Notes: For the original T5 pre-trained models[2], which were pre-trained with a mixture of unsupervised and supervised objectives, Adam or AdamW optimizers are enough to get good results.)

## Overview

The popular Adam[3] optimizer keeps two additional values for each parameter. One stores the momentum; one stores the exponentially smoothed squared gradients. Therefore, the memory requirement is tripled comparing to the vanilla SGD optimizer. Adafactor dramatically reduces this requirement (more than half) while retaining comparable performance (tested on the WMT ’14 En→De translation task with the classic transformer seq2seq architecture).

The authors of Adafactor firstly propose to **replace the full smoothed squared gradients matrix with a low-rank approximation**. This reduces the memory requirements for the square gradients from O(nm) to O(n+m).

Secondly, Adafactor removes momentum entirely. This causes some training instability. The authors think that the out-of-date second-moment accumulator (the exponential smoothing of the squared gradients) might be the cause. By **increasing the decay rate with time** (new values have higher importance) and **clipping the gradient update**, Adafactor can converge normally even without momentum.

Finally, Adafactor multiplies the learning rate by the scale of the parameters (this is called “relative step size”). The authors showed that training with relative step sizes provides more robustness to differently scaled embedding parameters.

## Factored Second Moment Estimation

Adafactor refactor the exponential moving average of squared gradients $V \in \mathbb{R}^{n \times m}$ to $RS$, where $R \in \mathbb{R}^{n \times 1}$ and $S \in \mathbb{R}^{1 \times m}$. It has an analytic solution for minimizing the I-divergence (generalized Kullback-Leibler divergence):

<div>$$d(p, q) = p\log\frac{p}{q} - p + q$$</div>

The solution only requires us to store the moving averages of the row sums and the column sums:

<div style="max-width: 300px; margin-left: auto; margin-right: auto;">{{< figure src="factorization.png" caption="Alg 1: The low-rank approximation of V" >}}</div>

Looking at the [corresponding part in the implementation](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L553), here's the part that update the moving average of the row and column sums:

```python
exp_avg_sq_row.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-1))
exp_avg_sq_col.mul_(beta2t).add_(1.0 - beta2t, update.mean(dim=-2))
```

And the implementation of the analytic solution (`rsqrt` means the reciprocal of the square root $1/\sqrt{input}$):

```python
@staticmethod
def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
    c_factor = exp_avg_sq_col.rsqrt()
    return torch.mm(r_factor.unsqueeze(-1), c_factor.unsqueeze(0))
```

The above corresponds to $1/\hat{V}_t = 1/(R_tCt/1^\intercal_nR_t)$. The $(1 - \beta^{t}_2)$ part (a.k.a. bias correction) in Alg 1 has been removed due to a reformulation of $\beta^{t}_2$ in the latter part of the paper.

## Removing Momentum

The authors demonstrated that fast decay of the second moment estimator has convergence problems, while slow decay has stability problems:

<div style="max-width: 300px; margin-left: auto; margin-right: auto;">{{< figure src="decay-1.png" caption="Table 1" >}}</div>

And the problem of slow decay is the larger-than-desired updates:

<div style="max-width: 350px; margin-left: auto; margin-right: auto;">{{< figure src="decay-2.png" caption="" >}}</div>

### Update Clipping

One of the proposed solutions is to clip the update according to the root-mean-square over all parameters in a weight matrix or vector:

<div>$$RMS(U_t) = RMS_{x \in X}(u_{xt}) = \sqrt{Mean_{x \in X}(\frac{g^2_{xt}}{\hat{v_{xt}}})}$$</div>

RMS is [implemented as a static method](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L484):

```python
@staticmethod
def _rms(tensor):
    return tensor.norm(2) / (tensor.numel() ** 0.5)
```

The tensor should already be unscaled (i.e., $\frac{g_{xt}}{\sqrt{v_{xt}}}$). The `.norm(2)` calculates the root sum squared, and `.numel() ** 0.5` convert it to the root mean squared.

The update then is [clipped accordingly to a threshold](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L569) $d$:

<div>$$\hat{U}_t = \frac{U_t}{max(1, RMS(U_t) / d)}$$</div>

```python
update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
update.mul_(lr)
```

It will effectively cap the unscaled update at $d$ (a horizontal line in Figure 1.).

### Increasing Decay Parameter

Another solution is to use an increasing $\beta_2$. The proposed family of schedules is:

<div>$$\hat{\beta}_{2t} = 1 - \frac{1}{t^c}, t \geq 1$$</div>

This can be [implemented in a one-liner](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L551):

```python
beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
```

Note that $\hat{\beta}_{2t}$ has been the reformulated, which eliminates the need to do bias correction.

## Relative Step Size

Adafactor multiplies the given learning rate by the scale of the parameters, which is defined as the root-mean-square of its components. Therefore, parameters with bigger values get bigger updates and vice versa:

<div>$$ \alpha_t = max(\epsilon_2, RMS(X_{t−1})) \rho_t $$</div>

The paper calls $\alpha_t$ the “absolute step size” and $\rho_t$ the “relative step size.”

The relative step size is implemented [here](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L474):

```python
if param_group["scale_parameter"]:
    param_scale = max(param_group["eps"][1], param_state["RMS"])
return param_scale * rel_step_sz
```

One crucial detail that one can easily get wrong (I did) is that the RMS is [calculated on a single parameter tensor](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L548)(a matrix or a vector). The learning rate is scaled by the magnitude of the entire weight matrix or vector, not the magnitude of a single parameter.

```python
state["RMS"] = self._rms(p_data_fp32)
```

Now we have the complete Adafactor algorithm:

<div style="max-width: 400px; margin-left: auto; margin-right: auto;">{{< figure src="adafactor.png" caption="" >}}</div>

## Confusing Parameter Naming

One problem of this implementation is the naming of its class parameters. There are three parameters that control the learning rate: `scale_parameter`, `warmup_init`, and `relative_step`. But in fact, only the first parameter — `scale_parameter` — implements the relative step size in the last section. The latter two only control the learning rate schedule.

With `relative_step=True` and `warmup_init=False`, the learning rate will be a simple inverse-square root decay used by the paper:

<div>$$\rho_t = min(10^{-2}, \frac{1}{\sqrt{t}})$$</div>

With `relative_step=True` and `warmup_init=True`, it adds a linear warmup stage to the schedule:

<div>$$\rho_t = min(10^{-6} \cdot t, \frac{1}{\sqrt{t}})$$</div>

They are implemented in the `_get_lr` [static method](https://github.com/veritable-tech/transformers/blob/8dcc2dfc2bdbd2e4838c7aa3a1e1775a0d23de5a/src/transformers/optimization.py#L469):

```python
rel_step_sz = param_group["lr"]
if param_group["relative_step"]:
    min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
    rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
```

As you can see, there's nothing to do with learning rate scaling by the magnitude of the parameter.

### Using Custom Learning Rate Schedule

Astute readers might already notice that when `relative_step=False` and `warmup_init=False`, the `rel_step_size` is simply the learning rate given the user has given to the optimizer. We can use regular PyTorch learning rate schedulers to control that variable. My [pull request](https://github.com/huggingface/transformers/pull/9751) fixed a bug that prevents the variable from being incorrectly updated by Adafactor.

## Working Examples

I've written [some code](https://github.com/ceshine/finetuning-t5) that fine-tunes T5 and mT5 models on NLI datasets using PyTorch Lightning. [This is where I set up the Adafactor optimizer](https://github.com/ceshine/finetuning-t5/blob/4ee255b3ed7c449ef5bd513cb8446c06f84708aa/mnli/train.py#L366):

```python
optimizer = Adafactor(
    self.model.parameters(),
    relative_step=False, warmup_init=False,
    clip_threshold=1.0, lr=self.config.learning_rate,
    scale_parameter=True
)
```

I used [a combination of linear warmup and cosine annealing](https://github.com/ceshine/finetuning-t5/blob/4ee255b3ed7c449ef5bd513cb8446c06f84708aa/mnli/train.py#L409) to schedule the learning rates:

```python
scheduler = {
    'scheduler': pls.lr_schedulers.MultiStageScheduler(
        [
            pls.lr_schedulers.LinearLR(optimizer, 0.0001, lr_durations[0]),
            CosineAnnealingLR(optimizer, lr_durations[1])
        ],
        start_at_epochs=break_points
    ),
    'interval': 'step',
    'frequency': 1,
    'strict': True,
}
```

I've published a [Kaggle notebook](https://www.kaggle.com/ceshine/preprocess-and-finetune-t5-1-1-full/) that fine-tunes the `google/t5-v1_1-base` model on the MultiNLI dataset and gets a competitive result. I've observed that my learning rate schedule performs better than the inverse-square root decay recommended by the paper.

An [mT5 version](https://www.kaggle.com/ceshine/pytorch-lightning-finetune-mnli-pretrained-mt5) that further fine-tunes an MNLI fine-tuned `google/mt5-base` model on a multi-lingual dataset is also available. Because of the low resource of the multi-lingual corpus, I froze the embedding matrix in this one to prevent overfitting.

## References

1. Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., … Raffel, C. (2020). [mT5: A massively multilingual pre-trained text-to-text transformer.](http://arxiv.org/abs/2010.11934)
2. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., … Liu, P. J. (2019). [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.](http://arxiv.org/abs/1910.10683)
3. Kingma, D. P., & Ba, J. L. (2015). [Adam: A method for stochastic optimization.](https://arxiv.org/abs/1412.6980)
