---
slug: kaggle-jigsaw-toxic-2019
date: 2019-08-04T00:00:00.000Z
title: "[Notes] Jigsaw Unintended Bias in Toxicity Classification"
description: "Bias Reduction; Summary of Top Solutions"
tags:
  - kaggle
  - machine_learning
  - deep-learning
  - nlp
keywords:
  - kaggle
  - deep learning
  - nlp
url: /post/kaggle-jigsaw-toxic-2019/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/sky-clouds-sunlight-dark-690293/)" >}}

# Preamble

Jigsaw hosted [a toxic comment classification competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview/evaluation)[2] in 2018, and has also created an API service for detecting toxic comments[3]. However, it has been shown that the model trained on this kind of datasets tend to have some biases against minority groups. For example, a simple sentence "_I am a black woman_" would be classified as toxic, and also more toxic than the sentence "_I am a woman_"[4]. This year's [_Jigsaw Unintended Bias in Toxicity Classification_ competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview)[1] introduces an innovative metric that aims to reduce such biases and challenges Kagglers to find out the best score we can get under this year's new dataset.

Unsurprisingly, fine-tuned BERT models[6] dominate the leaderboard and are reported as the best single model in top solutions. People get better scores by using custom loss weights, negative sampling techniques, and ensembling BERT with different kinds of (weaker, e.g. LSTM-based) models with ensemble schemes more appropriate with this metric. In my opinion, this competition doesn't produce some ground-breaking results, but is a great opportunity for data scientists to learn how to properly fine-tune pretrained transformer models and optimizing this peculiar metric that put an emphasis on minority groups.

XLNet[7] was published and its pretrained model released in the final week of the competition. Some top teams had already put in effort incorporating it into their ensembles. But I suspect the full potential of XLNet had not been achieved yet given such a short time.

## My Experience

I entered this competition with two weeks left. I built a pipeline that I am rather happy with and made some good progress at the start. Unfortunately, the metric function I copied from a public Kaggle Kernel was bugged (an important threshold condition `>= 0.5` was replaced by `> 0.5`), which severely undermines my effort in the last week because I was optimizing the wrong thing, and it caused the local cross-validation score to deviate from the public leaderboard score. In the end, I was placed at [187th](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/leaderboard) on the private leaderboard. I used only Kaggle Kernel and Google Colab to train my models.

After the competition, I spent some time debugging my code and finally found the bug. I published the corrected solution [on the on Github at **ceshine/jigsaw-toxic-2019**](https://github.com/ceshine/jigsaw-toxic-2019). It should be able to get into silver medal range by ensembling 5 to 10 models. I tried incorporating the loss weighting used in 6th place solution[11] and the "power 3.5 weighted sum" ensemble scheme used in 2nd place solution[8], and was able to reach the 70th-place private score with 5 _BERT-base-uncased_ and 2 _GPT-2_ models.

I think the performance of my single models can still be slightly improved, but the rest of the gains needed to reach gold medal range probably can only be achieved by creating a larger and more diverse ensemble. (Hopefully, I'll manage to find time to get back to this in the future.)

# Bias Reduction

So did the metric really reduced unintended bias? The following is a sample of cross-validation metric scores from [a single BERT-base-uncased model](https://www.kaggle.com/ceshine/bert-finetuning-public?scriptVersionId=17655093):

- Overall AUC: 0.972637
- Mean BNSP AUC: 0.965623 (background-negative subgroup-positive)
- Mean BPSN AUC: 0.928757 (background-positive subgroup-negative)
- Mean Subgroup AUC: 0.906707

However, these scores only show that the model is much better than a model that take random guesses. We need a human-level score baseline or the score from models trained using only overall AUC to have a proper comparison.

## Qualitative Research

One way to quickly find if the model still contains severe bias against minority groups is to do some quick qualitative checks. This is what we have previously, before the new metric:

{{< single_tweet 900867154412699649 >}}

(I think the above results were taken from the Perspective API[3]. The model used by the API appears to have been changed as the predicted toxicity are different now.)

Here's what we have now (the numbers are in percentage(%)):

{{< figure src="toxicity-1.png" caption="Results from [a fine-tuned BERT-base-uncased model](https://github.com/ceshine/jigsaw-toxic-2019/blob/774ed716f1323ab2987f2b43ece197975259b33c/notebooks/Model%20Diagnostics.ipynb)" >}}

Not bad! Only "I am a gay white man" has a toxicity probability larger than 20%. The new metric at least managed to eliminate these severe biases that look ridiculous to most human beings.

A few more examples from the actual (validation) dataset:

{{< figure src="toxicity-2.png" caption="Results from [a fine-tuned BERT-base-uncased model](https://github.com/ceshine/jigsaw-toxic-2019/blob/774ed716f1323ab2987f2b43ece197975259b33c/notebooks/Model%20Diagnostics.ipynb)" >}}

It shows that the model can correctly identify subgroups in more complicated sentences.

# Summary of Top Solutions

Currently only covers 2nd, 3rd, and 4th place solutions.

- Loss Weighting:
  - Simply count the number of appearances of an example in the four AUC scores that constitute the final metric score. [8]
  - Increase the weight of positive examples. [9]
  - Target weight multiplied by log(toxicity_annotator_count + 2). [9]
- Blending:
  - "Power 3.5 weighted sum"[16], take `prob ** 3.5` of the predicted probability before calculating the weighted sum/average when ensembling. [8]
  - Use [optuna](https://github.com/pfnet/optuna) to decide blending weights. [9]
- Target variables:
  - Discretize the target variable (via binning). [8]
  - Use identity columns as auxiliary targets. [8]
  - Use fine-grain toxicity columns as auxiliary targets. [10]
  - Mixing primary and auxiliary targets when making final predictions. (e.g. `toxicity_prediction - 0.05 * identity_attack_prediction`) [10]
- BERT language model fine-tuning. Some reported it successful[8][10], while some didn't[9]. It is resource-intensive and takes a lot of times to train. (I had also tried it myself with a subset of the training dataset, but did not get better downstream models.)
  - [8] also changed the segment_id when fine-tuning via `segment_ids = [1]*len(tokens) + padding`. (Not sure why.)
- Negative downsampling: cut 50% negative samples (in which target and auxiliary variables are all zero) after the first epoch. [9]
- Head + tail sequence truncation[17]. [9]
- Further fine-tuning with old toxic competition data. [9]
- CNN-based classifier head for GPT-2 models. [10]
- Pseudo-labeling for BERT: didn't work[9].
- Data augmentation: didn't work[9].
- Prediction value shift of each subgroup to optimize BPSN & BNSP AUC: didn't work[9].

# References

1. [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview)
2. [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview/evaluation)
3. [Perspective: What if technology could help improve conversations online?](https://perspectiveapi.com/#/home)
4. [Reported biased results (on Twitter)](https://twitter.com/jessamyn/status/900867154412699649)
5. [Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.](https://arxiv.org/abs/1903.04561)
6. [Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](http://arxiv.org/abs/1810.04805)
7. [Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le, Q. V. (2019). XLNet: Generalized Autoregressive Pretraining for Language Understanding.](http://arxiv.org/abs/1906.08237)
8. [[2nd place] solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661)
9. [[3rd place] solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471)
10. [[4th place] COMBAT WOMBAT solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100811)
11. [[6th place] Kernel For Training Bert (The real stuff)](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97487)
12. [[7th place] solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100611)
13. [[9th place] solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100530)
14. [[14th place] solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100821)
15. [[23rd place]A Loss function for the Jigsaw competition](https://www.kaggle.com/mashhoori/a-loss-function-for-the-jigsaw-competition)
16. [Reaching the depths of (power/geometric) ensembling when targeting the AUC metric](https://medium.com/data-design/reaching-the-depths-of-power-geometric-ensembling-when-targeting-the-auc-metric-2f356ea3250e)
17. [Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to Fine-Tune BERT for Text Classification?](http://arxiv.org/abs/1905.05583)

## My Solution / Implementation

- [Github repository — **ceshine/jigsaw-toxic-2019**](https://github.com/ceshine/jigsaw-toxic-2019)
- [Kaggle Training Kernel](https://www.kaggle.com/ceshine/bert-finetuning-public?scriptVersionId=17512842)
- [Kaggle Inference Kernel](https://www.kaggle.com/ceshine/toxic-2019-simple-ensemble-public?scriptVersionId=18261117)
- [Colab Training Notebook](https://colab.research.google.com/drive/1g0enYROgp7K6bOVSy9jmsPUg29ZVhGXs)
