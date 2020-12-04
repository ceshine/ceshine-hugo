---
slug: recsys-reproducibility
date: 2020-12-04T00:00:00.000Z
title: "[Paper] Are We Really Making Much Progress?"
description: "A Worrying Analysis of Recent Neural Recommendation Approaches"
tags:
  - recsys
keywords:
  - recsys
  - recommender
url: /post/recsys-reproducibility/
---

## Introduction

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/workers-people-boats-ships-pier-5708691/)" >}}

Today we’re examining this very interesting and alarming paper in the field of recommender systems — [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/abs/1907.06902). It also has an extended version still under review — [A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research](https://arxiv.org/abs/1911.07698).

The first author of the papers also gave an overview and answered some questions to the first paper in this YouTube video (he also mentioned some of the contents in the extended version, e.g., the information leakage problem):

{{< youtube JlLHIrzrmi4 >}}

## Key Points

### Identified Problems

1. Reproducibility: less than half of the top papers (7/18 in the original paper, 12/26 in the extended version) in this field can be reproduced.
2. Progress: only 1/7 (in the original paper) of the results can be shown to outperform well-tuned baselines consistently.

{{< figure src="table-2-summary.png" caption="[source: the YouTube presentation](https://www.youtube.com/watch?v=JlLHIrzrmi4)" >}}

The extended paper also raised some issues that make measuring the progress of the field harder:

1. Arbitrary experimental design: the choices of the evaluation procedures are not well justified.
2. Selection and propagation of weak baselines: as some of the top papers started to use only weaker baselines, the newer papers tend to keep using only those baselines, thus polluting the entire line of research.
3. Errors and information leakage: besides mistakes in code, some of the paper committed one of the cardinal sins in ML — use the test set to do hyper-parameter tuning.

### Implications for Practitioners

1. Don’t go with the fancy deep learning algorithm on the first try. It could cost you valuable time as the complexity of modern deep learning algorithms are often relatively high and still give you sub-optimal performance.
2. Instead, start by examining your dataset and fine-tuning applicable baseline models. If your dataset is highly unbalanced, personalized recommendations could be worse than simply recommending the most popular items.
3. Remember, the receivers of the recommendations are actual humans. The metric you choose does not necessarily reflect the users’ preference. Make sure improvement in your selected metric can translate to higher user satisfaction is your top priority.

## As a Guide to the Field

The author of the paper [published the code on Github](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation), which, combined with the paper, can be a great learning resource for practitioners. However, bear in mind that this paper only covers the top-n recommendation task. You might also need to find specialized models for other tasks.

{{< figure src="table-1-baseline-methods.png" caption="[source: the appendix of the extended paper](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/master/DL_Evaluation_TOIS_Additional_material.pdf)" >}}

{{< figure src="fig-1-neural-methods.png" caption="[source: the extended paper](https://arxiv.org/abs/1911.07698)" >}}
