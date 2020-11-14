---
slug: the-book-of-why
date: 2020-11-14T00:00:00.000Z
title: "The Book Of Why: The New Science of Cause and Effect"
description: "A Preliminary Book Review"
tags:
  - book
  - causal-inference
keywords:
  - book
  - causal-inference
url: /post/the-book-of-why/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/illustrations/cause-effect-causation-blame-5666661/)" >}}

## Impression

I just finished [The Book of Why by Judea Pearl](https://www.goodreads.com/book/show/36204378-the-book-of-why). This book is one of those that I wish I had picked it up a lot earlier. It makes a convincing case on what is missing in traditional probabilistic thinking and why the causal models can help to fill in the gap.

Although reading this book probably won't help you in finding a job as a data scientist or AI/ML engineer, but I genuinely think that every data scientist should read it to better understand the limitation of the current statistical learning methods. The model-free approaches to AI are unlikely to bring us Artificial General Intelligence(AGI). Blindingly throwing data at machine learning algorithms can only get us this far. (There already [seems to be some research in reinforcement learning](https://worldmodels.github.io/) that shows world models that imitate how humans perceive the world can help build more intelligent agents. However, I'm not yet an expert in reinforcement learning, so my interpretation can be wrong.)

I started reading another book by Pearl — [Causality: Models, Reasoning, and Inference](https://www.goodreads.com/book/show/174276.Causality) — a few years ago. It was a lot more technical than this one, and I dropped it shortly after because I found it too dry and not immediately useful to my work. In contrast, The Book of Why is a lot more beginner-friendly. The first half of the book is simply intellectually entertaining to read, establishing the foundation of Pearl's causal methodology. The second part describes the work by Pearl and colleagues to climb up the ladder of causation and is a bit more technical and mind-bending. However, the formulas are quite intuitive. As an example, the version of my book contains an error in the front-door adjustment formula, it uses a joint probability at a place of a conditional probability. I found it very counter-intuitive and tried to roughly derive the formula on my own using traditional probability axioms (I did not know anything about the do-calculus at this point). And still, I found that a conditional probability makes more sense. Googling the front-door adjustment formula and reading further in the book both confirmed my suspicion. It shows that Pearl's framework is very compatible with the human brain.

## Caveats

My view is likely to be biased because I have a degree in statistics. I suspect that people without statistics-101-level knowledge won't find reading the book as entertaining as I did. On the other hand, because this book is targeted at the general public, [some can find](https://www.goodreads.com/review/show/2644478614) that the skipping of technical details unforgivable. Maintaining a balance of accessibility and depth in science writing is hard. In my opinion, this book did an alright job, especially in the later chapters. It spares the readers of the tedious mathematical proofs (it did provide a brief proof to the front-door adjustment formula, though) and presents the most important results of the field and its applications. Readers can find a more technical book for a more formal introduction if they are intrigued. I am definitely going to find a recent technical book on this topic (probably [this one](https://www.goodreads.com/book/show/27164550-causal-inference-in-statistics)) to read.

[Some people find](https://www.goodreads.com/review/show/2401468531) Pearl's way to tell the story self-aggrandizing and unbearable. I can see their point, but I don't agree that it has reached a point that makes the book unreadable. Pearl is proud of his and his colleagues' work, and I totally respect that. I also find his work fascinating and share his excitement as he recounted their eureka moments.

One thing to bear in mind — **you won't be able to actually do causal modeling after reading this book**. You'll be able to handle situations very similar to the simple examples given in the book, but even in those, the book did not provide complete solutions. The book gives you a tour of the landscape of the field, so you know where to look if you need to do something. The rest is up to you.

## Conclusion

As someone who was trained in mostly classical and also a bit Bayesian statistics, this book brings a fresh perspective of how we can model the world. The frustration of statisticians wrestling with confounders and causal effects because of the lack of vocabulary in the statistics language strongly resonates with me. I give this book a 5/5 rating because of the impact it has on me (I'd give it a 3.5/5 if based purely on the style and presentation of writing).
