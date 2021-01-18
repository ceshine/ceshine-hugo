---
slug: trim-down-sentencepiece-vocabulary
date: 2021-01-18T00:00:00.000Z
title: "Reducing the SentencePiece Vocabulary Size of Pretrained NLP Models"
description: "Useful for fine-tuning on a subset of available languages"
tags:
  - nlp
keywords:
  - nlp
  - tip
url: /post/trim-down-sentencepiece-vocabulary/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/illustrations/cubes-cube-geometry-abstract-3381438/)" >}}

## Motivation

Q: Why and when would we want to trim down the vocabulary size of a pretrained model?

A: When a large portion of the vocabulary isn't used in your downstream task, it will make sense to get rid of the redundant part of the vocabulary to increase the model speed.

For example, Google's multilingual version of [T5](https://github.com/google-research/text-to-text-transfer-transformer) — [mT5](https://github.com/google-research/multilingual-t5) — was pretrained on 101 languages. Imagine if we only use English, Japanese, and Chinese in our downstream text generation task. We would waste a lot of time and space to process the rows in the embedding matrix and the LM head that corresponds to tokens that never appear in the dataset.

In this post, I'll demonstrate how to reduce the vocabulary size of a trained [**SentencePiece**](https://github.com/google/sentencepiece) model. SentencePiece [is used in XLNet, ALBERT, Marian, and T5](https://huggingface.co/transformers/tokenizer_summary.html#sentencepiece). Other types of tokenizers are not covered.

Specifically, we'll shrink the vocabulary of the `mt5-small` pretrained model. All tokens that are not used in the Chinese part of the [XNLI dataset](https://github.com/facebookresearch/XNLI) will be removed. As a result, the vocabulary size will go down from 250K to below 31K — an 87.6% reduction.

## References

The solution presented in this post comes from these two notebooks:

- [add new vocab](https://github.com/google/sentencepiece/blob/9cf136582d9cce492ba5a0cfb775f9e777fe07ea/python/add_new_vocab.ipynb) from google/sentencepiece
- [reduce vocab](https://github.com/bojone/t5_in_bert4keras/blob/6cf50dbf3ffd3b4e9f36a59ee9f98356cf686de0/tokenizer/reduce.py) from bojone/t5_in_bert4keras

I create an example showcasing the mechanism behind the code and verify the result. The complete notebook [can be accessed here on Github](https://github.com/ceshine/finetuning-t5/blob/8d4db99e11c0356db7c4535e9caaae723f656a51/notebooks/Manipulate%20Sentencepiece%20Vocabulary.ipynb).

## Code Walkthrough

### Download the Pretrained Model

We use [huggingface/transformers](https://github.com/huggingface/transformers) to download the pretrained tokenizer(SentencePiece model):

```python
from pathlib import Path
import shutil

from transformers import MT5Tokenizer

Path("cache/").mkdir(exist_ok=True)
if Path("cache/mt5-small").exists():
    shutil.rmtree("cache/mt5-small")

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
tokenizer.save_pretrained("cache/mt5-small")
```

### Download the XNLI dataset

Again, we use the [datasets library from huggingface](https://github.com/huggingface/datasets) to download the Chinese part of the XNLI dataset:

```python
from datasets import load_dataset

dataset = load_dataset("xnli", "zh")
```

### Collect the Tokens

Since we want to keep all the tokens that appear in this downstream dataset, we need to get a list of them:

```python
from itertools import chain
from tqdm import tqdm

def tokenize_data(data, batch_size=1024):
    global seen
    for i in tqdm(range(0, len(data), batch_size)):
        seen = seen.union(
            set(
              chain.from_iterable(
                tokenizer.batch_encode_plus(data[i:(i+batch_size)],
                return_attention_mask=False)["input_ids"]
              )
            )
        )

seen = set()
for subset in ("train", "test", "validation"):
    print(subset)
    tokenize_data(dataset[subset]["hypothesis"])
    tokenize_data(dataset[subset]["premise"])

# You can also add some additional (meta) tokens:
seen = seen.union(set(tokenizer.encode("mnli premise: hypothesis: <unk>")))
```

### Load the SentencePiece Model

Here we load the pretrained SentencePiece model into memory (as a Protocol Buffers object):

```python
from sentencepiece import sentencepiece_model_pb2 as model

m = model.ModelProto()
m.ParseFromString(open("cache/mt5-small/spiece.model", 'rb').read())

# There are some reserved places for speical tokens
for i, piece in enumerate(m.pieces[:320]):
    if i % 20 == 0:
        print(i, piece.piece)
```

We can see that the first 259 tokens are reserved for functional tokens. It might be a good idea to keep them.

### Shrink the SentencePiece Model

Because `m.pieces` is a Protocol Buffers field, we can not merely point it to a new list. Instead, we need to use the field's methods to manipulate its content:

```python
kept_pieces, i = [], len(m.pieces) - 1
while len(m.pieces):
    piece = m.pieces.pop()
    if i < 259 or i in seen:
        kept_pieces.append(piece)
    i -= 1
kept_pieces = list(reversed(kept_pieces))

# Backup the old model
Path("cache/mt5-small/spiece.model").rename("cache/mt5-small/spiece.model.old")
# Write the new model to disk
with open("cache/mt5-small/spiece.model", 'wb') as f:
    f.write(m.SerializeToString())
```

We'll also need to keep track of the tokens that are retained, so we can know which rows to keep in the embedding matrix and the LM head.

```python
import json

kept_ids = sorted(list(seen.union(set(range(259)))))
print(len(kept_ids))
with open("cache/mt5-small/kept_ids.json", 'w') as f:
    json.dump(kept_ids, f)
```

### Verification

We can verify that our new SentencePiece model can correctly tokenize our dataset (by encode and then decode sentences):

```python
import random
tokenizer = MT5Tokenizer.from_pretrained("cache/mt5-small")

for i in random.sample(range(100), k=10):
    # the space placements are slightly different from the original
    converted = tokenizer.decode(
        tokenizer.encode(dataset["train"]["hypothesis"][i]), skip_special_tokens=True
    ).replace(" ", "")
    assert converted == dataset["train"]["hypothesis"][i].replace(" ", "")
```

## Moving Further

Now we know how to create a new SentencePiece model with a smaller vocabulary size. The next step would be to use it to fine-tune an NLP model. **We'll need to modify the embedding matrix and the LM head** (if the LM head's weight and the embedding matrix are not tied together). It is beyond the scope of this post since the NLP model can be implemented in any of the deep learning frameworks (PyTorch, Tensorflow, MXNet, etc.).

Multilingual pretrained models are not the only use cases of this technique. It can also be useful when the upstream model is trained on a corpus covering several domains, while you are only interested in one of those domains in the downstream task. Just make sure you retain all the necessary tokens in the downstream task; otherwise, the performance might suffer.
