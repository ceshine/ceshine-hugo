+++

Categories = [ "Machine Learning", "Golang" ]
Description = "Using Go to solve Kaggle Avazu challenge by implementing FTRL-Proximal Algorithm"
Tags = []
date = "2014-12-09T17:42:55+10:00"
title = "Implement FTRL-Proximal Algorithm in Go - Part 1"

+++

For the sake of practicing, I've re-written [tinrtgu's solution](https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory) to the Avazu challenge on Kaggle using Go. I've made some changes to save more memory, but the underlying algorithm is basically the same. (See this [paper](https://gist.github.com/ceshine/c0f9538c48beb2069f57) from where the alogorithm came for more information).

The go code has been put on [Github Gist](https://gist.github.com/ceshine/c0f9538c48beb2069f57). Any constructive comments are welcomed on that gist page, as I haven't added a comment section on this blog. (I haven't even set up Google Analytics, so I have no idea how many people are reading thi blog) I'm also working on a concurrent version utilizing the built-in support of concurrency in Go. So theoretically it would run faster in multi-core environment.
