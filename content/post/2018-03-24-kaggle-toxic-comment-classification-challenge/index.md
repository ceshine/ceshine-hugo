---
slug: kaggle-toxic-comment-classification-challenge
date: 2018-03-24T05:40:19.985Z
title: "[Review] Kaggle Toxic Comment Classification Challenge"
subtitle: "A Preliminary Postmortem"
tags:
  - deep-learning
  - kaggle
  - nlp
keywords:
  - machine learning
  - deep learning
  - kaggle
  - nlp
url: /post/kaggle-toxic-comment-classification-challenge/
---

{{< figure src="18gwWxW96N8CSCpOVZZ1tbA.jpeg" caption="[Photo Credit](https://pixabay.com/en/pollution-toxic-products-environment-3075857/)" >}}

## Introduction

[Jigsaw toxic comment classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) features a multi-label text classification problem with a highly imbalanced dataset. The test set used originally was revealed to be already public on the Internet, so [a new dataset was released](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/47835) mid-competition, and the evaluation metric was [changed from Log Loss to AUC](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48639).

I tried a few ideas after building up my PyTorch pipeline but did not find any innovative approach that looks promising. Text normalization is the only strategy I had found to give solid improvements, but it is very time consuming. The final result (105th place/about top 3%) was quite fitting IMO given the time I spent on this competition(not a lot).

(There were some [heated discussion](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52574) around the topic of some top-ranking teams with Kaggle grand masters getting disqualified after the competition ended.)

Public kernel blends performed well in this competition (i.e. did not over-fit the public leaderboard too much). I expected it to overfit, but still selected one final submission that used the best public blend of blends to play it safe. Fortunately it paid off and gave me a 0.0001 boost in AUC on private leaderboard:

{{< figure src="1bpkp8RUQ8QYhHADiW7pdVA.png" caption="[Table 1: Selected submissions in descending order of private scores](https://docs.google.com/spreadsheets/d/1R3u6tXDa9CVByaRW9FY8FpxRTqwZsBWfLYuqC1gu8A8/edit?usp=sharing)" >}}

In this post, I’ll review some of the techniques used and shared by top competitors. I do not have enough time to test every one of them myself. Part 2 of this series will be implementing a top 5 solution by my own, if I ever find time to do it.

## List of Techniques

I tired to attribute techniques to all appropriate sources, but I’m sure I’ve missed some sources here and there. **_Not all techniques are covered because of the vast amount of contents shared by generous Kagglers_**. I might come back and edit this list in the near future.

- Translation as Augmentation (test time and train time) [1][2][13]
- Pseudo-labelling [1]
- Text normalization [3][4]
- Multi-level stacking — 4 levels [3]; 2 levels [11]; 1level stacking + 1 level weighted averaging [12]
- Non-English embeddings [2]
- Diverse pre-trained embeddings — Train multiple model separately on different embeddings
- Combine several pre-trained embeddings — concatenate after a RNN layer [3]; Concatenate at embedding level [5]; Variant of [Reinforced Mnemonic Reader](https://arxiv.org/abs/1705.02798) [7]
- When truncating texts, retain both head and tail of the texts [1]
- K-max pooling[6]
- Multiple-level of stacking [3]
- Deepmoji-style attention model [3]
- (NN) Additional row-level features:_“Unique words rate” and “Rate of all-caps words”_[4]
- (NN) Additional word-level feature(s): If a word contains only capital letters [4]
- (GBM) Engineered features other than tf-idf [12]
- BytePair Encoding [5][8]
- [R-net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) [7]
- CapsuleNet [11]
- Label aware attention layer [7]
- (20180413 Update) leaky information — small overlaps of IPs between training and testing dataset. One overlapping username. [14]

## Pseudo-labelling and Head-Tail Truncating

I’ve already tried these two techniques and trained a couple of models for each.

Head-tail truncating (keeping 250 tokens at head, 50 tokens at tail) helped only a bit for bi-GRU, but not for QRNN. It basically had no effect on my final ensemble.

For pseudo-labelling(PL), I used the test-set predictions from my best ensemble as suggested in [1], and they improved the final ensemble a little (see table 1). I’d assume that adding more model trained with PL will further boost the final AUC. However, the problem of this approach is the leakage it produces. The ensemble model had seen the the all the validation data, and that information leaked into its test set predictions. So the local CV will be distorted and not comparable to those trained without PL. Nonetheless, this technique does create the best single model, so it’ll be quite useful for production deployment.

I think the more conservative way of doing PL is to repeat the train-predict-train(with PL) process, so the model is trained twice for every fold. But that’ll definitely takes more time.

## References:

1. [The 1st place solution overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557) by [Chun Ming Lee](https://www.kaggle.com/leecming) from team _Toxic Crusaders_
1. [The 2nd place solution overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52612) by [neongen](https://www.kaggle.com/neongen) from team _neongen & Computer says no_
1. [The 3rd place solution overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52762) by [Bojan Tunguz](https://www.kaggle.com/tunguz) from team _Adversarial Autoencoder_
1. [About my 0.9872 single model](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644) by [Alexander Burmistrov](https://www.kaggle.com/mrboor) from team _Adversarial Autoencoder_
1. [The 5th place brief solution](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52630) by [Μαριος Μιχαηλιδης KazAnova](https://www.kaggle.com/kazanova) from team _TPMPM_
1. [Congrats (from ghost 11th team)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52526) by [CPMP](https://www.kaggle.com/cpmpml)
1. [The 12th place single model solution share](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52702) by [BackFactoryVillage](https://www.kaggle.com/goldenlock) from team _Knights of the Round Table_
1. [The 15th solution summary: Byte Pair Encoding](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52563) by [Mohsin hasan](https://www.kaggle.com/tezdhar) from team _Zehar 2.0_
1. [The 25th Place Notes (0.9872 public; 0.9873 private)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52647) by [James Trotman](https://www.kaggle.com/jtrotman) (solo)
1. [The 27th place solution overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52719) by [Justin Yang](https://www.kaggle.com/zake7749) from team _Sotoxic_
1. [The 33rd Place Solution Using Embedding Imputation (0.9872 private, 0.9876 public)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52666) by [Matt Motoki](https://www.kaggle.com/mmotoki) (solo)
1. [The 34th, Lots of FE and Poor Understanding of NNs [CODE INCLUDED]](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52645) by [Peter Hurford](https://www.kaggle.com/peterhurford) from team _Root Nice Square Error :)_
1. [A simple technique for extending dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038) by [Pavel Ostyakov](https://www.kaggle.com/pavelost)
1. [The 184th place write-up: code , solution and notes(without blend)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/54209) by [Eric Chan](https://www.kaggle.com/ericlikedata)
