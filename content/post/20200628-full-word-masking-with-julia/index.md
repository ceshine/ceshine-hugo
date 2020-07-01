---
slug: julia-whole-word-masking
date: 2020-06-28T00:00:00.000Z
title: "Using Julia to Do Whole Word Masking"
description: "Syntax almost as friendly as Python, while running up to 100x faster"
tags:
  - julia
  - nlp
keywords:
  - julia
  - nlp
url: /post/julia-whole-word-masking/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://unsplash.com/photos/IhsaTDKzdwg)" >}}

# Introduction

In my last post, [[Failure Report] Distill Fine-tuned Transformers into Recurrent Neural Networks](https://blog.ceshine.net/post/failed-to-distill-transformer-into-rnn/), I tried to distill the knowledge of a fine-tuned BERT model into an LSTM or GRU model without any data augmentation and failed to achieve satisfiable results. In the follow-up works, I tried to replicate the easies-to-implement augmentation method — masking — used in [1] and see its effect. The masking described in [1] is called “whole word masking” [2], that is, masking the whole word instead of just masking a single word piece.

It is non-trivial to implement whole word masking, as it would require the sampling process to be aware of which word piece is itself a whole word, and which is part of a word. As you may know, doing text processing in pure Python is quite slow comparing to other compiled languages. I recently picked up the Julia programming language, which promises the flexibility of scripting languages and the speed of compiled languages, and thought that it was a good opportunity to test Julia in the field.

This post describes the Julia code I wrote for this task and shows that for this specific task the Julia code is as simple to write as Python, while runs up to 100x faster than its pure Python counterpart.

## The Algorithm

This is the algorithm I used to do whole word masking (given that the examples are already tokenized to word pieces):

1. For each example, mark all the word pieces that are either a whole word or the first piece of a word (by using a mask).
2. Randomly sample N marked pieces for each example (N is a hyper-parameter).
3. Replacing the selected pieces with "[MASK]".
4. Check if the next piece is a part of this word (tokens start with "##" in BERT tokenizer). If so, also replace it with "[MASK]".
5. Repeat step 4 until the condition is false or the end of the example is reached.

# Benchmarks

Notebook used in this section:

- [Python](https://github.com/ceshine/transformer_to_rnn/blob/9b684e1204670adff17cd7770d009fa6c5569230/notebooks/03-1-3-benchmark-python.ipynb)
- [Julia](https://github.com/ceshine/transformer_to_rnn/blob/a593063fda951fedfa8a9a66f4ed51a690d81611/notebooks/03-1-2-benchmark-julia.ipynb)

## Summary

(Comparing the **_mean_** run time here as the `%timeit` magic doesn't provide the median run time.)

- Tokenizing examples: 15 seconds (shared by both Python and Julia Pipeline).
- Adding Special Tokens
  - Python: 42 ms (estimated)
  - Julia: 41 ms
- Marking First Pieces
  - Python: 326 ms
  - Julia: **47 ms** (single-threaded)
  - Julia: 39 ms (multi-threaded)
- Sample One Word to Mask
  - Python: 8.2 s (using `Numpy.random.choice`)
  - Julia: **69 ms**
- Masking
  - Python: 725 ms (copying the examples)
  - Julia: **426 ms** (copying the examples)
  - Python: 300 ms (estimated)
  - Julia: **10 ms**

### Remarks

- The most time-consuming part is tokenizing the examples. So in reality optimizing the tokenizer has the most potential (That's why huggingface has [re-implemented the word-piece tokenizers in Rust](https://github.com/huggingface/tokenizers)).
- But the eight seconds saved on sampling by switching to Julia is also a significant improvement, and just took a few lines to implement.
- Copying the examples takes around 300 to 500 ms, and is the most expensive operation besides tokenization. So try to avoid it if possible. (If you need the augment the same dataset multiple times, you have no choice to copy the examples.)

## Adding Special Tokens

A simple operation that adds "[CLS]" to the head and "[SEP]" to the tail. Python and Julia are equally fast in this one.

### Python

```python
def add_special_tokens(sentence):
    sentence.insert(0, "[CLS]")
    sentence.append("[SEP]")
tmp = deepcopy(sentences)
for sentence in tmp:
    add_special_tokens(sentence)
```

### Julia

```julia
function add_special_tokens!(sentence)
    pushfirst!(sentence, "[CLS]")
    push!(sentence, "[SEP]")
end
tmp = deepcopy(sentences)
results = add_special_tokens!.(tmp)
```

## Marking First Pieces

Create binary masks to filter out word piece that is not the first word piece of a word. Julia is starting to outperform Python.

### Python

```python
def is_first_piece(tokens):
    return [not token.startswith("##") for token in tokens]
first_piece_masks = [is_first_piece(sent) for sent in sentences]
```

### Julia

Vectorized (single-thread) version:

```julia
function is_first_piece(arr::Array{String,1})
    return .!startswith.(arr, "##")
end
results = is_first_piece.(sentences)
```

A multi-thread version is also provided, which can sometimes be faster depending on your hardware:

```julia
results = [Bool[] for _ in 1:length(sentences)]
Threads.@threads for i in 1:length(sentences)
    results[i] = is_first_piece(sentences[i])
end
```

## Sampling

Randomly sample one word from each example to be masked. Since I can't think of any simple way to vectorized this in Python, a naive for-loop approach is used. Vectorizing in Julia, on the other hand, is fairly straight-forward. As a result, the Julia version is vastly faster (100x) than the Python one.

Note: I used Numpy in the Python implementation, so it's not really "pure python" in this case.

### Python

```python
def sample(first_piece_masks, n=1):
    results = []
    for mask in first_piece_masks:
        if sum(mask) <= n:
            results.append([])
            continue
        probabilities = np.asarray(mask) / float(sum(mask))
        results.append(np.random.choice(np.arange(len(mask)), size=n, p=probabilities))
    return results
masking_points =  sample(first_piece_masks)
```

### Julia

```julia
using StatsBase
function sample_mask_position(first_piece_mask, n=1)
    if sum(first_piece_mask) <= n
        return Int64[]
    end
    return sample(1:length(first_piece_mask), Weights(first_piece_mask), n, replace=false)
end
masking_points = sample_mask_position.(first_piece_masks)
```

## Masking

Full word masking. This one inevitably has to use some loop to scan the example. For loops are not a problem for Julia, so the Julia version is much faster (30x) than Python.

The implementation presented here copies the examples inside the function so the original examples can be augmented multiple times.

### Python

```python
def masking(rows, first_piece_masks, masking_points):
    augmented_rows = deepcopy(rows)
    for idx in range(len(masking_points)):
        for pos in masking_points[idx]:
            augmented_rows[idx][pos] = "[MASK]"
            while pos +1 < len(first_piece_masks[idx]) and first_piece_masks[idx][pos + 1] == 0:
                pos += 1
                augmented_rows[idx][pos] = "[MASK]"
    return augmented_rows
augmented_sentences = masking(sentences, first_piece_masks, masking_points)
```

### Julia

```julia
function masking(rows::Vector{Vector{String}}, first_piece_masks::Vector{Vector{Bool}}, masking_points::Vector{Vector{Int64}})
    augmented_rows = deepcopy(rows)
    for idx in 1:length(masking_points)
        for pos in masking_points[idx]
            augmented_rows[idx][pos] = "[MASK]"
            while pos + 1 <= length(first_piece_masks[idx]) && first_piece_masks[idx][pos + 1] == 0
                pos += 1
                augmented_rows[idx][pos] = "[MASK]"
            end
        end
    end
    return augmented_rows
end
augmented_sentences = masking(sentences, first_piece_masks, masking_points)
```

# Conclusion

This is the first time I integrate Julia in an NLP pipeline, and the results are encouraging. The easy of development of Julia is on the same level as Python, but the is on a totally different level. In this example, the most improvement in speed comes from the sampling process, but it only represents less than 40 % of the total run time. And the total run time in Python is relatively short. I look forward to seeing what kind of speedup Julia can bring in bigger datasets or more complicated tasks.

([The notebook actually used in the pipeline](https://github.com/ceshine/transformer_to_rnn/blob/master/notebooks/03-1-1-masking-training-sequences.ipynb)).

# References

1. Tang, R., Lu, Y., Liu, L., Mou, L., Vechtomova, O., & Lin, J. (2019). [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks.](http://arxiv.org/abs/1903.12136)
2. [BERT: New May 31st, 2019: Whole Word Masking Models](https://github.com/google-research/bert)
