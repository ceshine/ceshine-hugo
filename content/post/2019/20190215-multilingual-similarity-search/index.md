---
slug: multilingual-similarity-search-using-pretrained-bidirectional-lstm-encoder
date: 2019-02-15T04:20:35.091Z
author: ""
title: "Multilingual Similarity Search Using Pretrained Bidirectional LSTM Encoder"
description: "Evaluating LASER (Language-Agnostic SEntence Representations)"
images:
  - featuredImage.jpeg
  - 1*oBOUeMjU0jbD1IyPLHPb5Q.png
  - 0*8KTX2sKSL9KNNCBG.png
tags:
  - machine_learning
  - deep-learning
  - pytorch
  - nlp
keywords:
  - machine-learning
  - deep-learning
  - pytorch
  - nlp
  - data-science
url: /post/multilingual-similarity-search/
---

{{< figure src="featuredImage.jpeg" caption="Photo by [Steven Wei](https://unsplash.com/photos/FITXkgVQJ9M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)" >}}

## Introduction

Previously I’ve demonstrated how to use pretrained BERT model to create a similarity measure between two documents in this post: **[News Topic Similarity Measure using Pretrained BERT Model](https://medium.com/the-artificial-impostor/news-topic-similarity-measure-using-pretrained-bert-model-1dbfe6a66f1d)**.

However, to find similar entries to* N* documents in corpus A of size _M_, we need to run *N*M\* feed-forwards. A more efficient and widely used method is to use neural networks to generate sentence/document embeddings, and calculate cosine similarity scores between these embeddings.

The [LASER (Language-Agnostic SEntence Representations) project](https://github.com/facebookresearch/LASER) released by Facebook provides a pretrained sentence encoder that can handle 92 different languages. Sentences from all languages are mapped into the same embedding space, so embeddings from different languages are comparable.

> Our system uses a single BiLSTM encoder with a shared BPE vocabulary for all languages, which is coupled with an auxiliary decoder and trained on publicly available parallel corpora. [1]

In this post we’ll try to reproduce the mapping between English and Chinese Mandarin sentences using the Tatoeba dataset created by [1]. After confirming we have the same results as reported in the paper, we’ll test if LASER can find the corresponding English titles to some (translated) articles from the New York Times Chinese version.

### A Big Caveat

LASER is licensed under [Attribution-NonCommercial 4.0 International,](https://github.com/facebookresearch/LASER/blob/master/LICENSE) so you can not do anything commercial with it.[ An update to the license seems to be upcoming](https://github.com/facebookresearch/LASER/issues/11), but not timetable is given yet.

You can write your own implementation, though. The training data used is publicly available (see appendix A in [1]). It takes 5 days and 16 V100 GPUs to train.

> Our implementation is based on [fairseq](https://github.com/pytorch/fairseq), and we make use of its multi-GPU support to train on 16 NVIDIA V100 GPUs with a total batch size of 128,000 tokens. Unless otherwise specified, we train our model for 17 epochs, which takes about 5 days. [1]

## Installation Notes

The core encoder itself only depends on PyTorch 1.0, but tokenization, BPE, similarity search requires some third-party libraries. Follow the [official installation instructions](https://github.com/ceshine/LASER#installation) to install tokenization scripts from Moses encoder and FastBPE. And install FAISS （a library for efficient similarity search and clustering of dense vectors） [via conda or from source](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).

I recommend using my fork of LASER since the original one requires you to install [\*transliterate](https://pypi.org/project/transliterate)\* package, which is only used for Greek, no matter you actually use Greek or not. My fork made it an optional dependency: **[ceshine/LASER](https://github.com/ceshine/LASER)**.

## Model Overview

{{< figure src="1*oBOUeMjU0jbD1IyPLHPb5Q.png" caption="Taken from [1]" >}}

The pretraining model architecture is not very different from the traditional sequence to sequence network. Several differences:

1. Bidirectional LSTM at every encoder layer.

1. Max pooling on top of the last encoder layer (instead of taking the hidden states at the last time step).

1. A linear transformation is performed on the pooled states, and then passed to the decoder to initialize the hidden states of its own LSTM units.

1. The pooled states are concatenated to the decoder input at every time step.

The sentence in the source language is encoded by the encoder, and then translated to the target language (English and Spanish) by the decoder. The encoder doesn’t know what the target language is. The target language is specified in the inputs (L*\_*id) to the decoder.

After pretraining, the encoder is extracted and used as-is (without any fine-tuning). It is proven to be quite useful in zero-shot transfer tasks (training data in one language, testing data in another).

{{< figure src="0*8KTX2sKSL9KNNCBG.png" caption="Left: monolingual embedding space. Right: shared embedding space. [Taken from [2]](https://code.fb.com/ai-research/laser-multilingual-sentence-embeddings/)." >}}

## Tatoeba English-Mandarin dataset

**[Tatoeba Notebook](https://github.com/ceshine/LASER/blob/master/notebooks/Tatoeba.ipynb)**.

There are 1,000 sentence pairs in English and Chinese Mandarin, stored in two text files `tatoeba.cmn-eng.eng` and `tatoeba.cmn-eng.cmn`.

Currently tokenization and BPE are basically shell commands wrapped in Python functions. We tokenize a text file by invoking `Token`:

```python
Token(
    str(DATA_PATH / "tatoeba.cmn-eng.cmn"),
    str(CACHE_PATH / "tatoeba.cmn-eng.cmn"),
    lang="zh",
    romanize=False,
    lower_case=True, gzip=False,
    verbose=True, over_write=False)
```

Then run BPE using `BPEfastApply`:

```python
bpe_codes = str(MODEL_PATH / "93langs.fcodes")
BPEfastApply(
    str(CACHE_PATH / "tatoeba.cmn-eng.eng"),
    str(CACHE_PATH / "tatoeba.cmn-eng.eng.bpe"),
    bpe_codes,
    verbose=True, over_write=False)
```

The pretrained encoder is loaded using the class `SentenceEncoder`:

```python
encoder = SentenceEncoder(
    str(MODEL_PATH / "bilstm.93langs.2018-12-26.pt"),
    max_sentences=None,
    max_tokens=10000,
    cpu=False)
```

The encoder consists of an embedding matrix (73640x320) and a 5-layer bidirectional LSTM module:

```python
Encoder(
  (embed_tokens): Embedding(73640, 320, padding_idx=1)
  (lstm): LSTM(320, 512, num_layers=5, bidirectional=True)
)
```

We compute the sentence embeddings and store it in yet another file:

```python
EncodeFile(
    encoder,
    str(CACHE_PATH / "tatoeba.cmn-eng.cmn.bpe"),
    str(CACHE_PATH / "tatoeba.cmn-eng.cmn.enc"),
    verbose=True, over_write=False)
```

And finally we create an FAISS index from that file containing embeddings:

```python
data_zh, index_zh = IndexCreate(
    str(CACHE_PATH / "tatoeba.cmn-eng.cmn.enc"), 'FlatL2',
    verbose=True, save_index=False)
```

A lot of temporary files are involved, which can be a bit annoying. It can definitely be improved, for example, by creating an end-to-end function hiding all the temporary file creation details from the user.

> similarity error en=>zh 4.10%
>
> similarity error zh=>en 5.00%`

The model yielded exactly the same error rate as reported in the paper[1]. Let’s take a look at some error cases. (The sentences are tokenized and BPE’ed.)

English to Chinese:

```
source:  i 'm at a loss for words .
predict: 我@@ 興@@ 奮 得 說 不 出@@ 話 來   。
correct: 我 不 知道 應@@ 該 說 什麼 才 好   。

source:  i just don 't know what to say .
predict: 我 不 知道 應@@ 該 說 什麼 才 好   。
correct: 我 只是 不 知道 應@@ 該 說 什麼 而@@ 已   ..@@ ....

source:  you should sleep .
predict: 你 应该 睡@@ 觉   。
correct: 你 應@@ 該 去 睡@@ 覺 了 吧   。

source:  so fu@@ ck@@ in ' what .
predict: 這@@ 是 什麼 啊   ？
correct: 那 又 怎@@ 樣   ?
```

Chinese to English:

```
source:  我 不 知道 應@@ 該 說 什麼 才 好   。
predict: i just don 't know what to say .
correct: i 'm at a loss for words .

source:  你 應@@ 該 去 睡@@ 覺 了 吧   。
predict: you should go to bed .
correct: you should sleep .

source:  那 又 怎@@ 樣   ?
predict: what is this ?
correct: so fu@@ ck@@ in ' what .

source:  我們 之@@ 間 已@@ 經 沒@@ 有 感@@ 情 了   。
predict: it would be better for both of us not to see each other any@@ more .
correct: i don 't like him any more than he lik@@ es me .
```

Most predictions in these cases are actually not far from the correct one. In some cases they are almost semantically identical (e.g., “you should go to bed” and “you should sleep”). The results were quite impressive.

## Chinese to English Mapping of Article Titles (the New York Times)

**[NYTimes Notebook](https://github.com/ceshine/LASER/blob/master/notebooks/New%20York%20Times%20Multilingual%20Titles.ipynb)**.

Similar to the [previous BERT post](https://medium.com/the-artificial-impostor/news-topic-similarity-measure-using-pretrained-bert-model-1dbfe6a66f1d), we use the RSS feeds from the New York Times to extract article titles. However, this time I used the Feedly API to read the RSS feeds, so you can try it on your end(no Feedly account is required):

```python
def fetch_latest(feed_url, count=500):
    res = requests.get(
        'https://cloud.feedly.com//v3/streams/contents'
        f'?streamId=feed/{feed_url}&count={count}')
    return res.json()
```

Since not every article from the NYT Chinese was translated from an English one, I downloaded the webpages and automatically extract the corresponding English titles if they exist.

The nearest 3 (English) neighbors were taken as the top 3 predictions:

```python
_, matched_indices = index_en.search(data_zh, 3)
```

And we got:

> Top 1 Accuracy: 47.37%
>
> Top 3 Accuracy: 57.89%

Around 50% accuracy may not look very good, but let’s take a look at error cases before jumping into conclusions:

```
Chinese:    美国将禁止中国设备进入5G市场
Correct:    Administration Readies Order to Keep China Out of Wireless Networks
Predict(1): Key Senator Warns of Dangers of Chinese Investment in 5G Networks
Predict(2): China Warns 2 American Warships in South China Sea
Predict(3): In 5G Race With China, U.S. Pushes Allies to Fight Huawei

--------------------

Chinese:    寻找艾衣提：抗议和诗歌的火种在维族音乐中燃烧
Correct:    MUSIC; In a Far-Flung Corner of China, a Folk Star
Predict(1): In China, Dolce & Gabbana Draws Fire and Accusations of Racism on Social Media
Predict(2): Album Review: Ariana Grande Is Living a Public Life. The Real Reveals Are in Her Music.
Predict(3): Ducking and Weaving: Corbyn’s Vanishing Act on Brexit

--------------------

Chinese:    劳工维权给习近平的“中国梦”蒙上阴影
Correct:    Workers’ Activism Rises as China’s Economy Slows. Xi Aims to Rein Them In.
Predict(1): Pessimism Looms Over Prospect of a Sweeping China Trade Deal
Predict(2): Wall Street Slides on Renewed U.S.-China Trade Fears
Predict(3): China’s Ambassador to Canada Blames ‘White Supremacy’ in Feud Over Arrests

--------------------

Chinese:    决定成功的“两种规则”
Correct:    The Two Codes Your Kids Need to Know
Predict(1): A Tale of Two Trumps
Predict(2): The Case Against ‘Border Security’
Predict(3): Personal Stories Behind the ‘Green Book’
```

The Chinese titles are mostly not the direct translation of the English one, so it’s understandable the encoder pretrained on translation tasks did not see them as almost the same. That being said, the predictions can be really off sometimes, as in the last case.

## Conclusions and Future Work

LASER provides a pretrained LSTM encoder that can take inputs from 92 languages (and close siblings in the language families) and map them to a shared embedding space.

The pretrained encoder itself already is quite useful in doing similarity search between sentences in different languages. The semantic features it can properly map are limited by the training corpus (see appendix A in [1]), though. The news corpus used in training, Global Voices, is relatively small. Some bizarre cases in the New York Times example can probably be attributed to this reason.

We did not evaluate zero-shot transfer tasks in this post because I haven’t thought of any interesting dataset I’d like to try on (except for the XNLI[4] and MLDoc[5] the paper had used [1].) I might write another post if I manage to find one.

## References

1. Mikel Artetxe and Holger Schwenk, [\*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)\* arXiv, 26 Dec 2018.

1. [Zero-shot transfer across 93 languages: Open-sourcing enhanced LASER library](https://code.fb.com/ai-research/laser-multilingual-sentence-embeddings/)

1. [Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to Sequence Learning with Neural Net- works.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

1. Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk and Veselin Stoyanov, [\*XNLI: Cross-lingual Sentence Understanding through Inference](https://aclweb.org/anthology/D18-1269)\*, EMNLP, 2018.

1. Holger Schwenk and Xian Li, [\*A Corpus for Multilingual Document Classification in Eight Languages](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf)\*, LREC, pages 3548–3551, 2018.
