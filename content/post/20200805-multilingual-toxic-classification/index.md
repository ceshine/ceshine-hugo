---
slug: multilingual-toxic-classification
date: 2020-08-05T00:00:00.000Z
title: "[Competition] Jigsaw Multilingual Toxic Comment Classification"
description: "The 3rd Jigsaw Text Classification Competition"
tags:
  - nlp
  - deep-learning
  - kaggle
keywords:
  - transformer
  - nlp
  - bert
  - deep learning
url: /post/multilingual-toxic-classification/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/5MZsCzjTut8)" >}}

# Introduction

[Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/) is the third Jigsaw toxic comment classification hosted on Kaggle. I've covered both [the first one in 2018](https://blog.ceshine.net/post/kaggle-toxic-comment-classification-challenge/) and [the second one in 2019](https://blog.ceshine.net/post/kaggle-jigsaw-toxic-2019/) on this blog. This time, Kagglers were asked to use English training corpora to create multilingual toxic comment classifiers that are tested in 6 other languages.

I've been taking a break from Kaggle during COVID pandemic, so I did not participate in this year's competition. However, reading top solutions is always very helpful whether you participated or not, and is exactly what I'm doing in this post. Due to time limitation, I will only cover a small part of the solutions shared. I'll update the post if I find the other interesting things later.

After reading some of the top solutions, it appears to me that there were limited novelty emerged from this competition. The model used by top teams are basically the same. What set them apart is some training details, ensemble techniques, and post-processing tricks. Maybe it's because the target task (binary classification) is relatively simple, as the 1st place team pointed out[1].

One thing I really like about this competition is that there are many well-written TPU notebooks shared publicly for both Tensorflow and PyTorch/XLA. It is especially helpful for us to catch up with the latest development of fast-evolving [PyTorch/XLA](https://github.com/pytorch/xla) project. I'll attach links to some of the notebooks in a later section.

# Key Components

## Dataset

We have the two English datasets from the previous two competition, and one validation dataset in **three** other languages. The test dataset contains three more languages than the validation dataset, in which we don't have any labeled data.

It turns out that auto-translated data[6] is better than no data in the target languages at all. All solution in the top 10 positions I've reviewed so far use some form of the auto-translated data. The 1st place solution[1] seems to only use the translated data, while others use both the original English data and the translated one[3][4][5].

On the other direction, the 6th place team[4] also found that translate the test corpus into English[6] and use English monolingual model to make prediction.

Other dataset like Open Subtitles can also be helpful[2].

## Pre-trained Models

Generally, [XLM-Roberta Large](https://arxiv.org/abs/1911.02116)[7] had been found to be the most effective single model for this competition.

Other monolingual pre-trained models had also been found to be valuable in the ensemble[1][2][4]. They are mostly BERT-based models.

## Pseudo Labelling

The 1st place team[1] find pseudo labelling essential in their success. They use pseudo labels to bootstrap the monolingual models, and then use the pseudo labels from the monolingual models to further improve the XLM-R models.

It seems that doing pseudo labelling on the multilingual model(XLM) does not work very well[4].

## Post-processing

Data in all six target languages are combined together to calculate in the AUC metric. This leaves some room for post-processing/optimization[1][3][5]. I don't really see how this would be helpful in real life. A more proper metric might be a weighed average of the individual AUC of each target language.

The 3rd place team[2] also found that 5% to 10% of the comments in the test dataset look like automatically generated, and they have a different target distribution. They capitalized on this finding and slightly improves the target metric.

There is also a team[4] that did not use any post-processing and still ended up okay.

# TPU Notebooks

## Tensorflow

- [My improved version](https://www.kaggle.com/ceshine/jigsaw-tpu-xlm-roberta) (post-competition, private 0.9402) of [the most popular Tensorflow notebook](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta).
- [[Colab] Train XLM-R large with TPU v2-8 on Colab ](https://www.kaggle.com/riblidezso/colab-train-xlm-r-large-with-tpu-v2-8-on-colab) (Reducing the system memory requirements.)

## PyTorch

- [[TPU-Training] Super Fast XLMRoberta](https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta)
- [I Like Clean TPU Training Kernels & I Can Not Lie](https://www.kaggle.com/abhishek/i-like-clean-tpu-training-kernels-i-can-not-lie)
- [XLM-Roberta-Large Pytorch TPU](https://www.kaggle.com/philippsinger/xlm-roberta-large-pytorch-pytorch-tpu?scriptVersionId=38462589)
- [Pytorch-XLA: Understanding TPU's and XLA](https://www.kaggle.com/tanulsingh077/pytorch-xla-understanding-tpu-s-and-xla)

# Reference

1. [1st place solution](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862) by team Lingua Franca
1. [3rd Place Solution](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160964) by team Ace Team
1. [4th place solution](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160980) by team rapids.nlp
1. [6th place solution](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/161095) by team toxu & nzholmes
1. [8th place solution](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160937) by team Psi
1. [List of Translated Datasets Links](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/159888)
1. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., … Stoyanov, V. (2019). [Unsupervised Cross-lingual Representation Learning at Scale.](http://arxiv.org/abs/1911.02116)
