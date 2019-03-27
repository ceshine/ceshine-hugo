+++
Categories = [ "Machine Learning" ]
Description = ""
Tags = ["machine_learning", "bayesian"]
date = "2015-07-11T17:54:38+10:00"
title = "Bayesian Logistic Regression using PyMC3"
+++

I've been reading this amazing (free) book [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers). I was half way through in early 2015, but dropped it because of some nuisances. But when I finally restarted reading it, I found it might be a good thing that I stopped reading for a while. Now I have more appreciation of the Bayesian methods and more mathematical understanding to fully grasp the idea the book trying to convey. (To be honest, I was quite confused about some concept like MAP in the first round of reading)

Although the book is based on PyMC2, I wanted to try the new PyMC3 and see how it performs. PyMC3 supports Python 3 and implements some cutting-edge algorithms. Wishing to find a balance between simplicity and real-world applicability, I decided to try fitting a Bayesian logistic regression model on the popular Kaggle Titanic dataset.

The [corresponding IPython notebook](http://nbviewer.ipython.org/gist/ceshine/c9a4308384e744f062f5) is on _nbviewer_. I borrow a large chunk of code from the official examples. It is still quite primitive. A lot of rooms for improvement, e.g., I didn't treat _PClass_ as a categorical variable. But it already gives a okay score on the leaderboard.

Interestingly, the inclusion of _Age_ feature drags down the score a little bit. The reason might be the large presence of missing values, or just simply I did something wrong. I might need to go back to review that.

Hopefully I'll try a PyMC2 version of this script and refine it a bit more. Also I'm aware that PyMC3 actually provides a shortcut for generalized linear models. Would love to try that as well.
