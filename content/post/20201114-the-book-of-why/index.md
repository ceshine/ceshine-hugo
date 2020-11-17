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

## Update (20201117)

For a different perspective, read Andrew Gelman's (who, incidentally, was a student of Donald Rubin) [review on The Book of Why](https://statmodeling.stat.columbia.edu/2019/01/08/book-pearl-mackenzie/). His review was one of my motivations to read The Book of Why, just so I can understand what the article is saying.

Gelman agrees with Pearl and Mackenzie that both qualitative modeling and quantitative modeling are important, and thinks that the examples in the book are quite engaging (as do I):

> And that brings me to the examples in the book. These are great. I find some of the reasoning hard to follow, and Pearl and Mackenzie’s style is different from mine, but that’s fine. The examples are interesting and they engage the reader—at least, they engage me—and I think they are a big part of what makes the book work.

However, as an academic statistician, he finds that some of the characterizations of statistician and social scientist by Pearl and Mackenzie are unfair:

> As noted above, Pearl and Mackenzie have a habit of putting down statisticians in a way that seems to reflect ignorance of our field.
> ...
> ...In any case, I find it unfortunate that they feel the need to keep putting down statisticians and social scientists. If they were accurate in their putdowns, I’d have no problem. But that’s not what’s happening here. Kevin Gray makes a similar point here, from the perspective of a non-academic statistician.

He pointed out some of the inaccurate comments on statistical methods in the book. Some of them I am not familiar enough to comprehend, but I did find the remark on Latin square in the book not making much sense.

I guess my professors in statistics would be disappointed by me not finding Pearl's characterization of statistician grossly offending. There indeed tends to be some oversimplification and exaggeration in Pearl's writing. But I do feel that the statistics education I received focused too much on quantitative methods (with possibly one exception for the data visualization course, which was brilliant), and I appreciate the simplicity and effectiveness of causal diagrams in specifying our prior belief to the world. It's possible that I did not pay enough attention in past courses to notice that the professors did try to convey the ways to understand causal effects correctly. Nonetheless, Pearl's approach is refreshing to me.

I'm currently reading Pearl's [Causal Inference in Statistics: A Primer](https://www.goodreads.com/book/show/27164550-causal-inference-in-statistics). It's as the title says, an introductory textbook, and is fairly easy to read. I only have the counterfactual chapter left before finishing the book. It is a very complementary read for The Book of Why. A lot of technical definitions and theorems that are left out in the book can be found here. However, I must admit that the "do" operator still does not fully make sense to me (Gelman doesn't think it makes sense as a general construct). I get that it's removing the incoming edges to the variable in the causal diagram, but I failed to find a way to build the intuition or explaining it coherently in plain English (the explanations I've seen so far all are too obscure in my view).

I'm interested to learn more about other approaches to causal inference by statisticians, some of them mentioned in Gelman's article. Maybe it's true that statisticians have already developed methods to handle causal relationships more formally, but I barely see any mention of causal relationships in the field of machine learning.

For example, a video recommender based on a nonlinear model may recommend a video to someone because their friends like it, but "my friends like it" and "I like it" may be confounded by "this video promotes a conspiracy theory". We can use tools such as [SHAP values](https://blog.ceshine.net/post/shap/) to estimate how much the conspiracy theory part contributed to the recommendation. However, the estimates can be inaccurate when such confounding exists, and we need a succinct way to raise and answer the question “is the reason behind this recommendation is really because it promotes a conspiracy theory, not because my friends like it?” (Don't get me wrong. The question is probably somewhat answerable by analyzing the model or plotting the relationship between the two covariates, but it'll be nice to have a diagram and a system of methods to help us do that.)
