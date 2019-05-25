+++
title = "[Notes] SHAP Values"
description = "A Unified Approach to Interpreting Model Predictions"
date = 2019-05-23T11:29:49+08:00
tags = [""]
images = ["cover.jpg"]
draft = false
url= "/post/shap/"
+++

{{< figure src="cover.jpg" caption="[Photo Credit](https://pixabay.com/photos/tai-qi-activity-body-fitness-1583805/)" >}}

Unlike other feature importance measures, SHAP values are fairly complicated and theoretically grounded. I kept forgetting the small details of how SHAP values works. These notes aim for making sure I understand the concept well enough and be something that I can refer back to once in a while. Hopefully it will also be helpful to you.

# Classic Shapley Value Estimation

*Shapley regression values*:

<div>$$\phi_{i} = \sum_{S \subset F \backslash \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)]$$</div>

> *Shapley regression values* are feature importances for linear models **in the presence of multicollinearity**. [1]

*Multicollinearity* means that predictor variables in a regression model are highly correlated. The existence of multicollinearity violates [one of the assumptions of multiple linear regression](https://www.statisticssolutions.com/assumptions-of-multiple-linear-regression/), and the regression coefficients cannot be reliably interpreted as importances.

Shapley regression values can be broken into three parts: the summation, combinatorial weight, and the part inside the square bracket. My preferred way to build the intuition is to read from right to left.

## Compare the predictions

<div>$$f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)$$</div>

Given a feature subset `$S \subset F$`, where `$F$` is the set of all features. A model `$f_S(x_S)$` is trained with features in `$S$` present, and another model `$f_{S \cup \{i\}}(x_{S \cup \{i\}})$` is trained with an additional feature *i* in interest. Then the above calculate the difference in model predictions from these two models.

This basically translates to the change in model prediction when a feature *i* is added.

## Assign weights

<div>$$\frac{|S|!(|F|-|S|-1)!}{|F|!}$$</div>

We can reformulate the above as:

<div>$$\frac{1}{|F|}\frac{1}{{|F|-1}\choose{|S|}}$$</div>

## Summation

Firstly, since we have `${|F|-1}\choose{|S|}$` different subsets of features with size `|S|`, their weights sums to `${1}/{|F|}$`.

All the possible subset sizes range from 0 to `$|F| - 1$` (we have to exclude the one feature we want its feature importance calculated). That is `$|F|$` different subset sizes. Combined with the previous result, the weights of all possible feature subsets sum to exactly **one**.

One important aspect of the weighting scheme is that it puts equal weights to all subset sizes. For example, the average model prediction difference when subset size is 2 are equally important as the one when subset size is 1. (I still have not developed an intuition of the reason of this equal weight assignment.)

## Shapley sampling values

There are `$2^{|F|}$` possible feature subsets, and we usually don't want to actually train `$2^{|F|}$` models just to get feature importances. Shapley sampling values approximate the effect of removing a variable from the model by **integrating over samples from the training dataset**. This idea later becomes a central part of SHAP value estimation.

Another technique shapley sampling values employs is to apply sampling approximations to the Shapley regression values. This technique is also used in the model-agnostic approximation *Kernel SHAP*, but the implementation details are not discussed in the paper. Instead we can read [the Python implementation of *Kernel SHAP*](https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py) to find out. (SHAP for tree models can be calculated without sampling approximation, as described in [2]).

# Theoretical Properties

Additive feature attribution methods use a mapping function `$x = h_x(x')$` to create a simplified inputs `$x'$`, and create a linear function of binary variables `$g(z') \approx f(h_x(z'))$`:

<div>$$g(z') = \phi_0 + \sum_{i=1}^{M}\phi_{i}z_{i}'$$</div>

It has been proved that the Shapley value is the only one possible additive feature attribution method that has the following three properties.

## Property 1 (Local accuracy)

<div>$$f(x) = g(x')$$</div>

The explanation model `$g(x')$` gives the same value as the original model `$f(x)$`. (Remember that `$x=h_x(x')$`.)

## Property 2 (Missingness)

<div>$$x'_i = 0 \Rightarrow \phi_i = 0 $$</div>

This is basically saying that when a feature is missing, the importance measure `$\phi$` of this feature should be zero (meaning no impact to the model).

## Property 3 (Consistency)

If a feature *i* has higher impact in model A than model B, then the importance measure `$\phi_i$` in model A should always be larger than the one in model B.

This property is arguably the most wanted property in explanation models. It is needed so that we can trust the calculated feature importance.

# SHAP (SHapley Additive exPlanation) values

The theoretical properties of Shapley values all sounds great and nice to have, but in reality we most likely won't be able to calculate the exact Shapley values. We have to use some approximation techniques, and their quality determines who well the theoretical properties will hold. The paper [1] provided some user study experiments to show that the SHAP values are more consistent comparing to other commonly used algorithms.

SHAP values makes one major approximation and two optional approximations under feature independence and model linearity assumptions. We'll cover them in the following sections.

## Conditional expectation function

Since most model cannot handles arbitrary pattern of missing inputs, the function `$f(z_S)$` is approximated by a conditional expectation `$E[f(x) | z_S]$`. This comes from *"integrating over samples from the training data set"* used in Shapley sampling values. When the training data set is too big, we use a smaller weighted background data set instead. The Python implementation recommends running a K-mean algorithm to create such background data sets.

We can accurately calculate this expectation when the model is a tree model. The training sample counts that goes through the left and right nodes are the weights of their respective predictions.

## Feature Independence

In many other models we are not able to efficiently calculate the expectation while taking the feature dependence into account. We have to assume feature independence and integrate samples from the background data set blindly without considering the values of other features.

## Model Linearity

Instead of integrating samples, we can directly fill in the **mean value of the feature in the background data set** if we assume the independent features and linear model.

# Kernel SHAP (Linear LIME + Shapley values)

Kernel SHAP is a model-agnostic approximation that expands the framework of LIME[3]. The benefits of this approach is that it doesn't require the evaluation of `$2^{|M|}$` expectations, with `$|M|$` being the number of features. Instead it takes only some samples and use a (weighted) linear regression and try to make it recapitulate the Shapley values.

Like in LIME, this approximation breaks the local accuracy and/or consistency, but the improved Shapley kernel has much better approximation accuracy according to the paper.

The weighting kernel in LIME is heuristically chosen, while this Shapley kernel is analytically chosen.

The sampling scheme is a bit mysterious, though. In LIME, the size of feature subset is uniformly sampled, and then the features are uniformly selected[3]. However, in most case we can just use the Python implementation without worrying this detail.

# Other model-specific approximations

**Deep SHAP (DeepLIFT + Shapley values)** builds on DeepLIFT, which can be seen as approximating SHAP values with feature independence and
linearity assumptions. This requires understanding the original DeepLIFT, therefore was not covered here.

**SHAP values for Tree Ensembles**[2] introduces an efficient algorithm that estimating SHAP values in polynomial time (`$O(TLD^2)$`) instead of exponential time (`$O(TL2^M)$`). And also a novel SHAP interaction values that capture pairwise interaction effects.

# References

1. [Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.](http://arxiv.org/abs/1705.07874)
2. [Lundberg, S. M., Erion, G. G., & Lee, S.-I. (2018). Consistent Individualized Feature Attribution for Tree Ensembles.](https://arxiv.org/abs/1802.03888)
3. [Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier.](http://arxiv.org/abs/1602.04938)