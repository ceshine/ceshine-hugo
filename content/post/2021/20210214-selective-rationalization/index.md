---
slug: selective-rationalization
date: 2021-02-14T00:00:00.000Z
title: "[Paper] Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control"
description: "Building competitive self-explaining NLP models"
tags:
  - nlp
  - interpretability
  - pytorch
  - paper
keywords:
  - nlp
  - interpretability
  - pytorch
  - paper
  - research
url: /post/selective-rationalization/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/fe2Lk4jDEio)" >}}

## Introduction

Model interpretability is crucial if we want to use AI models to make high-stake decisions (e.g., making medical diagnoses, preventing suicides, etc.). In NLP, one common way to get interpretability is to extract information from the trained models. For example, some use gradient-based input attribution techniques, some perturb the input to get explanations, and some use influence functions to find the most influential training examples to this particular input sequence. Another way is to make the model intrinsically explainable (e.g., a decision tree).

Selective rationalization creates self-explaining models without specialized designs nor architectural choices for the base model. This paper — “Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control”[1] — shows that we can get the self-explaining models with accuracy on par with the black-box ones with the proposed modification to the selective rationalization technique[2] (i.e., introspective extraction and complement control).

## Selective Rationalization

The goal of selective rationalization is to extract only the portion of the text relevant for prediction. The original design[2] consists of two players — a generator (that generates binary masks) and a predictor (that makes predictions based on the masked text).

<div style="max-width: 300px; margin-left: auto; margin-right: auto;">{{< figure src="arch-lei.png" caption="The cooperative framework from [2]. <br/> Source: [1]" >}}</div>

Because the predictor can only see the extracted(masked) text, we can be 100% sure that the dropped tokens do not influence the prediction. (Note that we cannot be sure that 100% of the extracted text is relevant to the prediction, though. Nevertheless, we can increase the confidence by making the binary mask sparser.)

{{< figure src="examples-1.png" caption="Examples of extracted rationales of questions in the AskUbuntu domain. Source: [2]" >}}

## Complement Control

> Degeneration refers to the situation where, rather than finding words in X that explain Y, R attempts to encode the probability of Y using trivial information, e.g., punctuation and position. [1]

<div style="max-width: 500px; margin-left: auto; margin-right: auto;">{{< figure src="degeneration.png" caption="An example of rationales extracted by different models. Source: [1]" >}}</div>

To avoid degeneration (model taking shortcuts), the paper proposed that we train another predictor (the complement predictor) based on the dropped text. If we can still get decent accuracy from the complement predictor, we likely have degeneration at hand.

<div style="max-width: 250px; margin-left: auto; margin-right: auto;">{{< figure src="complement-predictor.png" caption="Complement predictor.<br/>Source: [1]" >}}</div>

The paper proposes that we penalize the model when the cross-entropy from the (actual) predictor is within `h` to the one from the complement predictor:

<div>$$L_p = max\{L_p - L_c + h, 0\}$$</div>

## Introspective Extraction

The paper posits out that since the generator in [2] typically has **no direct access** to the outcome it aims to justify, the learning process may converge to a poorly performing solution. The solution they proposed is to include the predicted class (_**from a pretrained model**_) in the input to the generator. In this scheme, the generator would know which label in advance the selected rationales will be used to predict.

<div style="max-width: 250px; margin-left: auto; margin-right: auto;">{{< figure src="introspective.png" caption="Introspective generator.<br/>Source: [1]" >}}</div>

(_Notes from Ceshine_) The reason for the emphasis on “no direct access” is that the generator actually has indirect access to the label via the cross-entropy of the predictor it tries to minimize (we'll see more details in the next section). The generator has been implicitly predicting to which label this piece of text belongs. We can see the introspective extraction as a way to relieving the generator of this task and focus just on selecting the tokens.

## Training Generators

### Loss Function

This is the loss function for the generator:

<div>$$\min_{z(\cdot)} L_p + \lambda_s L_g + \lambda_s L_s + \lambda_c L_c$$</div>

We've covered `$L_g$` in the Complement Control section. `$L_s$` controls **sparsity** (_penalizes the model when the number of the selected tokens exceeds a threshold_), and `$L_c$` controls **continuity** (_penalizes the model when the number of gaps between selected tokens exceeds a threshold_).

### Policy Gradient

Because the token mask is binary, we cannot directly use gradient descent to train the generator. Instead, the paper uses policy gradient[3] to train the generator.

Policy gradient is commonly used in reinforcement learning[4]. As the paper references [3], I think what they implement is the [REINFORCE](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce) (Monte-Carlo policy gradient) algorithm. Since we have only one step in this design, it probably means sampling tokens according to the generated probabilities and taking averages.

(_Notes from Ceshine_) To have bounded rewards for training stability, the paper replace negative losses `$L_p$` and `$L_c$` with accuracy. I think clipped (negative) cross-entropy should have the same effect while supporting soft labels at the same time.

## Implementations

The paper provides [an official implementation of the algorithm on Github](https://github.com/Gorov/three_player_for_emnlp). However, the code is a bit dated (it depends on PyTorch 0.3.0). I've found a modern implementation at [interpretml/interpret-text](https://github.com/interpretml/interpret-text), which also provides [an example notebook](https://github.com/interpretml/interpret-text/blob/master/notebooks/text_classification/text_classification_introspective_rationale_explainer.ipynb).

Maybe I'll write a part 2 that analyzes the implementation if I find it helpful enough. Stay tuned.

## References

1. Yu, M., Chang, S., Zhang, Y., & Jaakkola, T. S. (2019). [Rethinking cooperative rationalization: Introspective extraction and complement control.](https://arxiv.org/abs/1910.13294)
2. Lei, T., Barzilay, R., & Jaakkola, T. (2016). [Rationalizing neural predictions.](https://arxiv.org/abs/1606.04155) EMNLP 2016 - Conference on Empirical Methods in Natural Language Processing, Proceedings, 107–117.
3. Willia, R. J. (1992). [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 8(3), 229–256.](https://doi.org/10.1023/A:1022672621406)
4. Lilian Weng (2018). [Policy Gradient Algorithms.](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce)
