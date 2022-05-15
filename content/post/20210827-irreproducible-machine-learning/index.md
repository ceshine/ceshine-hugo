---
slug: irreproducible-machine-learning
date: 2021-08-27T00:00:00.000Z
title: "[Notes] (Ir)Reproducible Machine Learning: A Case Study"
description: "A reminder of how easy it is to screw up when doing applied ML research"
tags:
  - machine-learning
  - dataset
keywords:
  - machine learning
  - dataset
url: /post/irreproducible-machine-learning/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/mountains-alps-mountaineering-cold-4695049/)" >}}

I just read this (draft) paper named “(Ir)Reproducible Machine Learning: A Case Study” ([blog post](https://reproducible.cs.princeton.edu); [paper](https://reproducible.cs.princeton.edu/irreproducibility-paper.pdf)). It reviewed 15 papers that were focusing on predicting civil war and evaluated using a train-test split. Out of these 15 papers:

1. **12** shared the complete code and data for their results.
2. **4** have errors.
3. **9** do not have hypothesis testing or uncertainty quantification (including 3 of 4 papers with errors).

Three of the papers with errors shared the same dataset. Muchlinski et al.[1] created the dataset, and then Colaresi and Zuhaib Mahmood[2] and Wang[3] reused the dataset without noticing the critical error in Muchlinski et al.'s dataset construction process — data leakage due to **imputing the training and test data together**.

The remaining paper — Kaufman et al.[4] — **include proxy variables in the prediction model**, which also introduced data leakage.

Two of these papers (Wang[3] and Kaufman et al.[4]) also committed the “crime” of using _k_-fold cross-validation with temporal data. Again, this created data leakage.

For the readers who are not familiar with the term, [data leakage](<https://www.wikiwand.com/en/Leakage_(machine_learning)>) is when the model training process uses information that will not be available at prediction time.

For example, if we want to predict a person's height as an adult right after they are born. Using the size of their shirts as an adult as a covariate/independent variable is using a proxy variable.

If some of the weights at birth are missing, using the test data to create an impute model would also create leakage. The impute model may see that one of the people in test data was weighted 3.5kg and is 1.75m tall. It would use that information when filling missing values and tilt the prediction value for that person towards 1.75m.

Finally, if we have two cohorts of data — one from 1900 and one from 2000. Mixing them together and employ _k_-fold validation would make the results much less reliable. The average height back in 1900 would be much lower than in 2000. In this hypothetical scenario, we collect height data once a century. Correctly stratifying the data when doing validation would show that it is very hard to predict the heights as an adult of infants born 100 years later. But usually, the line is more blurry. If the data is collected once a year, the damaging effect of using _k_-fold validation would not be as severe. It would be a subject of debate about whether it is appropriate to use _k_-fold.

This study also suggests that we should provide some uncertainty quantification, such as reporting bootstrapped confidence intervals for model performance. I've seen deep learning papers that train huge models provided the standard error of the model performances. If they can do that, there's no excuse for us not to give the uncertainty information for much smaller models and datasets.

Overall, it is a very thought-provoking paper. The authors emphasized that the purpose is not to bash the authors who wrote the erroneous papers but to make us aware of the fact that applied machine learning research is still the wild west. We should exercise caution and be mindful of common pitfalls when doing work in this field.

## References

1. David Muchlinski et al. “Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data”. en. In: Political Analysis 24.1 (2016), pp. 87–103.
2. Michael Colaresi and Zuhaib Mahmood. “Do the robot: Lessons from machine learning to improve conflict forecasting”. en. In: Journal ofPeace Research 54.2 (Mar. 2017), pp. 193–214.
3. Yu Wang. “Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data: A Comment”. en. In: Political Analysis 27.1 (Jan. 2019), pp. 107–110.
4. Aaron Russell Kaufman, Peter Kraft, and Maya Sen. “Improving Supreme Court Forecasting Using Boosted Decision Trees”. en. In: Political Analysis 27.3 (July 2019), pp. 381–387.
