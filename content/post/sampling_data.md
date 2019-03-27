+++
Categories = [ "Data Engineering" ]
Description = "Taking random samples from a large set"
Tags = ["data_eng"]
date = "2015-01-22T10:42:55+10:00"
title = "Random Sampling at the Command Line"
+++

When you receive a large dataset to analyze, you'd probably want to take a look at the data before fitting any models on it. But what if the dataset is too big to fit into the memory?

One way to deal with this is to take much smaller random samples from the dataset. You'll be able to have some rough ideas about what's going on. (Of course you cannot get global maximum or minimum through this approach, but that kind of statistics can be easily obtained in linear time with minimal memory requirements)

Two command-line utilities I've found quite useful: **shuf** and **[sample](https://github.com/jeroenjanssens/data-science-at-the-command-line/blob/master/tools/sample)**(from [Data Science at the Command Line](http://datascienceatthecommandline.com/))

**shuf** can be used to randomly get N record from the population:

```bash
    shuf -n 100 training_data.csv
```
**sample** can be used to randomly extract N% of the data:

```bash
    sample -r 0.3 training_data.csv
```
These are really easy examples, but should suffice in most cases. You can check out the documentation yourself and get creative if you feel like it.
