+++

Categories = [ "Machine Learning", "Golang" ]
Description = "Concurrent version of the previous implementation of FTRL-Proximal Algorithm"
Tags = []
date = "2014-12-09T17:42:55+10:00"
title = "Implement FTRL-Proximal Algorithm in Go - Part 2"

+++

I've actually finished the concurrent version of the algorithm a while ago, right after the previous post. Unfortunately my laptop broke and it took almost a month to repair.

Now I finally get to publish the [result](https://gist.github.com/ceshine/f7f93046c58fe6ee840b) here. I know that the code is not elegant nor properly documented, but it's a start. You'll need to set the **core** variable in the main function to the number of cores of your CPU. The program will simultaneously trains a number of models according to that value, and predict the average of the prediction from each model.

There are a bunch of things to be improved. Firstly, the distribution of the training data to each trainer is not really random. Secondly, cross validation doesn't work since we are not able to separate the data into two part for training and validation. So the cross validation value is just for reference.

Despite all the flaws, this algorithm already can get a decent score for the competition given some tuning. I've read that feature engineering is really important for this competition. I lost some time when my laptop is away, but I might get into that. I'll publish the code after the competition if this approach worked out.
