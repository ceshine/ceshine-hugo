---
slug: statistical-inference-bad-science
date: 2019-11-09T00:00:00.000Z
title: "[Notes] “Statistical Inference Enables Bad Science; Statistical Thinking Enables Good Science”"
description: "The Article by Christopher Tong on The American Statistician Volume 73, 2019"
tags:
  - science
  - statistics
keywords:
  - science
  - statistics
url: /post/statistical-inference-bad-science/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/road-night-monochrome-highway-4598095/)" >}}

[This article by Christopher Tong](https://www.tandfonline.com/doi/full/10.1080/00031305.2018.1518264) has got a lot of love from people I followed on Twitter, so I decided to read it. It was very enlightening. But to be honest, I don't fully understand quite a few arguments made by this article, probably because I lack the experience of more rigorous scientific experiments and research. Nonetheless, I think writing down the parts I find interesting and put it into a blog post would be beneficial for myself and other potential readers. Hopefully, it makes it easier to reflect on these materials later.

This article argues that instead of relying on the statistical inference on an isolated study, we should use guide scientific research of all kinds by statistical thinking, and validate claims by replicating and predicting finds in new data and new settings.

> Replicating and predicting findings in new data and new settings is a stronger way of validating claims than blessing results from an isolated study with statistical inferences.

Let's see the reasoning behind this claim.

# Introduction

First, Tong makes clear what “statistical inferences” are:

> Statistical inferences are claims made using probability models of data generating processes, **intended to characterize unknown features of the population(s) or process(es) from which data are thought to be sampled**. Examples include estimates of parameters such as the population mean (often attended by confidence intervals), hypothesis test results (such as p-values), and posterior probabilities.

Some of the widely used tools in statistical inference has come under fire recently for being misused or abused (as in [The ASA's Statement on p-Values: Context, Process, and Purpose](https://amstat.tandfonline.com/doi/full/10.1080/00031305.2016.1154108))

> Among these criticisms, McShane and Gelman (2017) succinctly stated that **null hypothesis testing “was supposed to protect researchers from over-interpreting noisy data. Now it has the opposite effect.”**

Tong tries to distinguish **exploratory** and **confirmatory** objectives of a study. He argues that most scientific research tends to be exploratory and flexible, but the statistical inference is only suitable in a confirmatory setting where study protocol and statistical model are fully prespecified.

> We shall argue that these issues stem largely from the _Optimism Principle_ (Picard and Cook 1984)that is an inevitable byproduct of the necessarily flexible data analysis and modeling work that attends most scientific research.

And the lack of this distinction in the current use of inferential methods in science has enabled biased statistical inference and encouraged a _Cult of the Isolated_ Study that short-circuits the iterative nature of research.

# Statistical Inference and the Optimism Principle

As Efron and Hastie stated in their new book "Computer-Age Statistical Inference: Algorithms, Evidence, and Data Science":

> It is a surprising, and crucial, aspect of statistical theory that the same data that supplies an estimate can also assess its accuracy.

I had similar doubts when receiving traditional statistics education. The bias and variance tradeoff is mentioned but it is generally up to use to decide where to draw the line. The cross-validation is clearly a more principled and objective approach. (See the [Breiman's classic paper “Statistical Modeling: The Two Culture”](http://www2.math.uu.se/~thulin/mm/breiman.pdf))

As Harrell, F. E., Jr. (2015) observed:

> Using the data to guide the data analysis is almost as dangerous as not doing so.

Essentially, when researchers devise their analysis approach based on the data, it creates a chance to overfit the data. Simmons, Nelson, and Simonsohn (2011) called these opportunities **_researcher degrees of freedom_**, and when abused to fish for publishable p-values, p-hacking.

> The resulting inferences from the final model tend to be biased, with uncertainties underestimated, and statistical significance overestimated, a phenomenon dubbed the **_Optimism Principle_** by Picard andCook (1984).

In extreme cases, nonsense data can still seem to make sense.

> In other words, it is possible to obtain a seemingly informative linear model, with decent R^2 and several statistically significant predictor variables, from data that is utter nonsense. This finding was later dubbed “**_Freedman’s paradox_**” (Raftery, Madigan, and Hoeting 1993).

This kind of bias would lead to an underestimation of the uncertainty because we picked the model that has fit the training data best.

> Chatfield (1995) used the term **_model selection bias_** to describe the distorted inferences that result when using the same data that determines the form of the final model to also produce inferences from that model.

# Exploratory and Confirmatory Objectives in Scientific Research

> The obvious way to avoid the difficulties of overfitting and produce valid statistical inferences is to completely prespecify the study design and statistical analysis plan prior to the start of data collection. T

Tong uses the phased experimentation of medical clinical trials as an example of scientific research where exploratory/confirmatory distinction is clearly made.

> This framework helps to separate therapeutic exploratory (typically Phase II) with therapeutic confirmatory (typically Phase III) objectives.

It doesn't prevent the expensive clinical dataset to be used for further exploratory work — to generate hypotheses for further testing in later experiments.

> A succinct perspective on such inferences is given by Sir Richard Peto, often quoted (e.g., Freedman 1998) as saying “you should always do subgroup analysis and never believe the results.”

And it doesn't mean the result from exploratory studies shouldn't be published.

> If the result is important and exciting, we want to publish exploratory studies, but at the same time make clear that they are generally statistically underpowered, and need to be reproduced.

# From the Cult of the Isolated Study to Triangulation

> The treatment of statistical inferences from exploratory research as if they were confirmatory enables what Nelder (1986) called **_The Cult of the Isolated Study_**, so that
> The effects claimed may never be checked by painstaking reproduction of the study elsewhere, and when this absence of checking is combined with the possibility that the original results would not have been reported unless the effects could be presented as significant, the result is a procedure which hardly deserves the attribute ‘scientific.

Simple replication is usually not sufficient. Tong uses the Wright Brothers as a demonstrative example.

> Munafo and Davey Smith (2018) define triangulation as “the strategic use of multiple approaches to address one question. Each approach has its own unrelated assumptions, strengths, and weaknesses. **Results that agree across different methodologies are less likely to be artifacts.**”

The notorious example of the report by the OPERA collaboration shows the importance of triangulation to uncover systematic errors.

> A particular weakness of the Isolated Study is that systematic errors may contaminate an entire study but remain hidden if no further research is done.

# Technical Solutions and Their Deficiencies

> The most widely known class of such methods is based on adjusting for multiple inferences. These range from the simple Bonferroni inequality to the modern methods of false discovery rate and false coverage rate (e.g., Dickhaus 2014).

> A second class of methods incorporates resistance to overfitting into the statistical modeling process, often through an optimization procedure that penalizes model complexity, an approach sometimes called regularization.

Tong also indicates that random splitting is still not the perfect solution.

> Unfortunately, such procedures (or their variants) are still vulnerable to the Optimism Principle, because random splitting implies that “left-out” samples are similar to the “left-in” samples (Gunter and Tong 2017).

So it's better to collect more data to overcome model uncertainty:

> Obtaining “more than one set of data, whenever possible, is a potentially more convincing way of overcoming model uncertainty and is needed anyway to determine the range of conditions under which a model is valid” (Chatfield 1995).

Tong also discussed another widely advocated solution — _model averaging_. Those who are familiar with Kaggle competitions should already have a firm grasp on this.

> Only through the iterative learning process, using multiple lines of evidence and many sets of data, can systematic error be discovered, and model refinement be continually guided by new data.

# More Thoughtful Solutions

> One strategy requires preregistering both the research hypotheses to be tested and the statistical analysis plan prior to data collection, much as in a late-stage clinical trial (e.g., Nosek et al. 2018).

However, the fact that most scientific research cannot fit the above paradigm is a big problem. A more realistic approach is _preregistered replication_.

> A variation on this theme is preregistered replication, where a replication study, rather than the original study, is subject to strict preregistration (e.g., Gelman 2015). A broader vision of this idea (Mogil andMacleod 2017) is to carry out a whole series of exploratory experiments without any formal statistical inference, and summarize the results by descriptive statistics (including graphics) or even just disclosure of the raw data.

# Enabling Good Science

Tong adapts a taxonomy of statistical activity by Cox (1957) and Moore (1992):

- Data production. The planning and execution of a study (either observational or experimental).
- Descriptive and exploratory analysis. Study the data at hand.
- Generalization. Make claims about the world beyond the data at hand.

> The first step of statistical thinking is to understand the objective of the study, its context, and its constraints, so that planning for study design and analysis can be fit for purpose.

## Data Production

> Feller (1969) pronounced that “The purpose of statistics in laboratories should be to save labor, time, and expense by efficient experimental designs” rather than null hypothesis significance testing.

Tong discusses a few experiment design techniques that should already be familiar to those who have taken formal statistics education. He also raises some practical concerns when conducting the experiment and its analysis.

> Data acquisition and storage systems should have appropriate resolution and reliability. (We once worked with an instrument that allowed the user to retrieve stored time series data with a choice of time-resolution. Upon investigation, we found that the system was artificially interpolating data, and reporting values not actually measured, if the user chose a high resolution.)

And other research degrees of freedom that is related to decisions around experiment design:

> Other researcher degrees of freedom can affect study design and execution. An instructive example for the latter is the decision to terminate data collection. Except in clinical trials, where this decision is tightly regulated and accounted for in the subsequent analysis (e.g., Chow and Chang 2012), many researchers have no formal termination rule, stopping when funding is exhausted, lab priorities shift, apparent statistical significance is achieved (or becomes clearly hopeless), or for some other arbitrary reason, often involving unblinded interim looks at the data.

## Data Description

> Moses (1992)warned us that
> Good statistical description is demanding and challenging work: it requires sound conceptualization, and demands insightfully organizing the data, and effectively communicating the results; not one of those tasks is easy. To mistakenly treat description as ‘routine’ is almost surely to botch the job.

Theory of Description:

> Mallows (1983) provided an interesting perspective on a Theory of Description. He noted that “A good descriptive technique should be appropriate for its purpose; effective as a mode of communication, accurate, complete, and resistant.”

Something like Tukey's (1977) five number summary (the minimum, first quartile, median, third quartile, and maximum) can be helpful to describe the variability of the data.

> Though we might not quantify uncertainty using probability statements, we can attempt to convey the observed variability of the data at hand, while acknowledging that it does not fully capture uncertainty... However, the use of such data summaries is not free of assumptions (e.g., unimodality, in some cases symmetry), so they are descriptive only in relation to these assumptions, not in an absolute sense.

## Disciplined Data Exploration

> Accord- ing to Tukey (1973), exploratory analysis of the data is not “just descriptive statistics,” but rather an “actively incisive rather than passively descriptive” activity, “with a real emphasis on the discovery of the unexpected.”

An example of how exploratory analysis may be essential for scientific inquiry is in the detection of and adjustment for batch effects.

> Leek et al. (2010) defined batch effects as “sub-groups of measurements that have qualitatively different behavior across conditions and are unrelated to the biological or scientific variables in a study.”

Tong also cites the warning of Diaconis (1985) about the danger of undisciplined exploratory analysis.

> If such patterns are accepted as gospel without considering that they may have arisen by chance, he considers it magical thinking, which he defines as“our inclination to seek and interpret connections and events around us, together with our disinclination to revise belief after further observation.”

## Statistical Thinking

> Statistical thinking begins with a relentless focus on fitness for purpose (paraphrasing Tukey 1962: seeking approximate answers to the right questions, not exact answers to the wrong ones), sound attitudes about data production and its pitfalls, and good habits of data display and disciplined data exploration.

Statistical thinking also involves a keen awareness of the pitfalls of data analysis and its interpretation, including:

- The correlation versus causation fallacy.
- The distinction between interpolation and extrapolation.
- The distinction between experimental and observational data.
- Regression to the mean.
- Simpson’s paradox, and the ecological fallacy.
- The curse of dimensionality

## Discussion

> There is no scientifically sound way to quantify uncertainty from a single set of data, in isolation from other sets of data comprising an exploratory/learning process.
> This brings to mind an observation made about certain research in materials science: “Even if the studies had reported an error value, the trustworthiness of the result would not depend on that value alone” (Wenmackers and Vanpouke 2012).
> By emphasizing principles of data production, data description, enlightened data display, disciplined data exploration, and exposing statistical pitfalls in interpretation, there is much that statisticians can do to ensure that statistics is “a catalyst to iterative scientific learning” (Box 1999).
