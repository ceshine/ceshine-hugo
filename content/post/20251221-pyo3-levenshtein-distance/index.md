---
slug: pyo3-levenshtein-distance
date: 2025-12-21T00:00:00.000Z
title: "Speed Up Your Python Scripts with Rust: A Levenshtein Distance Case Study"
description: "Discover how PyO3 enables Python developers to easily harness Rust's performance for critical path operations - 50x faster with clean integration."
tags:
  - rust
  - open-source
  - python
keywords:
  - python
  - rust
  - pyo3
url: /post/pyo3-levenshtein-distance/
---

Disclaimer: A 50x speedup is not guaranteed. Actual performance depends on the nature of the dataset and the hardware on which the code is run. Please refer to the Benchmarks section below for more information.

## Introduction

Recently, I finally found some time to learn the Rust programming language. I find its memory safety guarantee quite elegant, although it comes with the trade-off of a steep learning curve, especially when it comes to Rust’s ownership and lifetime system. It is very appealing to someone like me, whose primary tool is a scripting language and who writes low-level code only from time to time. Writing C/C++ code can easily lead to unstable runtime behavior or unexpected results in such circumstances.

For practice, I translated one of my CLI tools from Julia to Rust with the assistance of GPT-5-Codex. The tool performs a video-understanding task via a [LitServe](https://github.com/Lightning-AI/LitServe) HTTP server. I used the [Tokio library](https://docs.rs/tokio/latest/tokio/index.html) to run the three stages (frame extraction, frame inference, and post-processing) concurrently. In my opinion, Rust is much better suited to this kind of task than Julia. I’m also satisfied with the modularity and clarity of the new Rust codebase.

However, because I still mostly use Python for my various projects, making Rust code work in Python would offer the best of both worlds and empower me to write more efficient and maintainable code. Incidentally, I was studying the [fenic](https://github.com/typedef-ai/fenic) library and found that it had written some Polars extensions in Rust and integrated them into the main package to speed up certain compute-intensive tasks (e.g., handling JSON strings). I took the opportunity to study the tooling around Polars extensions and learned that [Polars](https://github.com/pola-rs/polars) and some other popular Rust-powered Python libraries (e.g., [Tokenizers](https://github.com/huggingface/tokenizers)) also use the same tooling (they usually have more sophisticated build configurations than fenic, though).

The tooling consists of [PyO3](https://pyo3.rs/v0.27.1/index.html)[1] and [Maturin](https://www.maturin.rs/tutorial.html)[2]. PyO3 is a Rust library that provides a bridge between Rust and Python, while Maturin is a build system that simplifies the process of building and distributing Rust-based Python packages. To test out the tooling, I decided to build a [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)[3] calculation function in Rust that can be imported and run in Python code. Getting to a working version was much simpler than I expected, and the speedup was immediately visible. However, getting it to a more robust and publishable state took me a bit more digging. This blog post will walk you through the process and highlight some key configurations and concepts that may confuse beginners.


## References

1. [PyO3 - Python Functions](https://pyo3.rs/v0.27.1/function.html)
2. [Maturin Tutorial](https://www.maturin.rs/tutorial.html)
3. [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
4. [GitHub Issue: Add dynamic = \["version"\] to pyproject.toml](https://github.com/PyO3/maturin/issues/1772)
