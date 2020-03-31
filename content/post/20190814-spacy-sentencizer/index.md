---
slug: spacy-sentencizer
date: 2019-08-14T00:00:00.000Z
title: "Customizing Spacy Sentence Segmentation"
description: "Making the Default Model More Robust by Add Custom Rules"
tags:
  - nlp
  - spacy
  - python
  - tips
keywords:
  - nlp
  - spacy
url: /post/spacy-sentencizer/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/cat-sweet-kitty-animals-feline-323262/)" >}}

# The Problem

Often in natural language processing(NLP), we would want to split a large document into sentences, so we can analyze the individual sentences and the relationship between them.

Spacy's pretrained neural models provide such functionality via their syntactic dependency parsers. It also provides a rule-based [_Sentencizer_](https://spacy.io/api/sentencizer), which will be very likely to fail with more complex sentences.

While the statistical sentence segmentation of spacy works quite well in most cases, there are still some weird cases on which it fails. One of them is the difficulty in handling the **’s** tokens, which I noticed when using Spacy version _1.0.18_ and model `en_core_web_md` version _2.0.0_.

For example, given this sentence (the title of a news article from The Atlantic):

> Hong Kong Shows the Flaws in China’s Zero-Sum Worldview.

Spacy returns three sentences:

1. Hong Kong Shows the Flaws in China
2. ’s
3. Zero-Sum Worldview.

Another example taken from a news article from the New York Times:

> Police officers fired tear gas in several locations as a day that began with a show of peaceful defiance outside the headquarters of China’s military garrison descended into an evening of clashes, panic and widespread disruption.

Spacy splits it into two sentences:

1. Police officers fired tear gas in several locations as a day that began with a show of peaceful defiance outside the headquarters of China
2. ’s military garrison descended into an evening of clashes, panic and widespread disruption.

The problem seems to be somewhat alleviated in the latest _2.1.0_ model, but still, the solution provided below will be helpful.

# The Solution

According to [Spacy's documentation](https://spacy.io/usage/linguistic-features#sbd-custom), we can add custom rules as a _custom pipeline component_ (before the dependency parser) that specifies the sentence boundaries. The later dependency parser will respect the `Token.is_sent_start` attribute set by this component.

We want to make sure that **’s** tokens will never be the start of a sentence. Here is how to do it:

{{< gist ceshine 839505744f7c144dbe24ff61c1393b06 >}}

Now Spacy will correctly identify the previous two examples as full sentences.

**_20190822 Update_**: Added rules that improves the handling of curly quotes.

# Source Code

{{< gist ceshine 7741974cf14d838c7a3b3e2c1031d8c7 >}}
