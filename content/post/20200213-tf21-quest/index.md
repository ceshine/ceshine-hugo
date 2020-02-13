---
slug: tf21-quest
date: 2020-02-13T00:00:00.000Z
title: "Tensorflow 2.1 with TPU in Practice"
description: "Case Study: Google QUEST Q&A Labeling Competition"
tags:
  - nlp
  - tensorflow
  - kaggle
keywords:
  - nlp
  - tensorflow
  - tpu
  - kaggle
url: /post/tf21-quest/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/the-side-of-the-road-snow-mountains-4259510/)" >}}

# Executive Summary

- **Tensorflow has become much easier to use**: As an experience PyTorch developer who only knows a bit of Tensorflow 1.x, I was able to pick up Tensorflow 2.x using my spare time in 60 days and do competitive machine learning.
- **TPU has never been more accessible**: The new interface to TPU in Tensorflow 2.1 works right out of the box in most cases, and greatly reduces the development time required to make a model TPU-compatible. Using TPU drastically increases the speed of experiment iterations.
- **We present a case study of solving a Q&A labeling problem by fine-tuning RoBERTa-base model from _huggingface/transformer_ library**:
  - [Codebase](https://github.com/ceshine/kaggle-quest)
  - [Colab TPU training notebook](https://gist.github.com/ceshine/752c77742973a013320a9f20384528a1)
  - [Kaggle Inference Kernel](https://www.kaggle.com/ceshine/quest-roberta-inference/data?scriptVersionId=28553401)
  - [High-level library TF-HelperBot](https://github.com/ceshine/tf-helper-bot/) to provide more flexibility than the Keras interface.
- (Tensorflow 2.1 and TPU are also very good fit for CV applications. A case study of solving a image classification problem will be published in about a month.)

# Acknowledgement

I was granted free access to Cloud TPUs for 60 days via Tensorflow Research Cloud. It was for the [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) competition. I chose to do this simpler [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge) competition first, but unfortunately couldn't find enough time to go back and do the original one (sorry!).

I was also granted \$300 credits for the [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) competition, and had used those to develope a [PyTorch baseline](https://github.com/ceshine/kaggle-tf2qa). They also covered the costs of Cloud Compute VM and Cloud Storage used to train models on TPU.

# Introduction

Google was handing out free TPU access to competitors in the [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) competition, as an incentive for them to try out the newly added TPU support in Tensorflow 2.1 (then RC). Because the preemptible GPUs on GCP are barely usable at the time, I decided to give it a shot. It all began with this tweet:

{{< single_tweet 1210092823564832768 >}}

Turns out that the Tensorflow model in [huggingface/transformers](https://github.com/huggingface/transformers) library can work with TPU without modification! I then proceeded to develop models using Tensorflow(TF) 2.1 for a simpler competition [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge).

I missed the post-processing trick in the QUEST competition because I spent most of my limited time wrestling with TF and TPU. After applying the post-processing trick, my final model would be somewhat competitive at around 65th place (silver medal) on the final leaderboard. **The total training time of my 5-fold models using TPUv2 on Colab was about an hour**. This is a satisfactory result in my opinion, given the time constraint.

## Tensorflow 2.x

The Tensorflow 2.x has become much more approachable, and the customized training loops provide a swath of opportunities to do creative things. I'm more confident now that I'll be able to re-implement top solutions of the competition in TF 2.x without banging my head on the door (at least less frequently).

On the other hand, TF 2.x is still not as intuitive as PyTorch. The documentation and community support still has much to be desired. Many of the search results still point to TF 1.x solutions which are not applicable to TF 2.x.

As an example, I ran into this problem in which the CuDNN failed to initialize:

{{< single_tweet 1227289969896505344 >}}

One of the solution is to limit the GPU memory usage, and here's a confusingly long thread on how to do so:

{{< single_tweet 1227289971368685568 >}}

## TPU Support

Despite of all the drawbacks, the TPU support in TF 2.1 is fantastic, and has become my main reason of using TF 2.1+. It's hard to imagine how the extremely unstable TPU support in Keras has evolved into this good piece of engineering.

Although Tensorflow Research Cloud gave my access to multiple TPUs, I only use one of them as I didn't see the need to do serious hyper-parameter optimization yet. The competition data set is not ideal for TPU, as it is quite small (a few thousands of examples). I have to limit the batch size to achieve the best performance (in terms of the evaluation metric), but it is still a lot faster than training on my single local GTX 1070 GPU (4 ~ 8x speedup). TPUv2 is more than sufficient in this case (comparing to TPUv3).

One potentially interesting comparison would be using two V100 GPUs, which combined are a little more expensive than TPUv2, with a bigger batch size to train the same model.

**The TPU Google Colab now also supports TF 2.1**. You are able to train models much faster with it than any of the free GPU Colab provides (currently the best offer is a single Tesla P100). Check this notebook for a concrete example:

{{< single_tweet 1215182582083538944 >}}

(I know that PyTorch has [its own TPU support](https://github.com/pytorch/xla) now, but it is still quite hard to use last time I check, and it is not supported in Google Colab. Maybe I'll take another look in the next few weeks.)
