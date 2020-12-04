---
slug: analyzing-tweets-with-r
date: 2018-02-27T00:07:25.807Z
title: "Analyzing Tweets with R"
description: "Based on “Text Mining with R” Chapter 7"
images:
  - 1*HrUUCqfdKxgodgzTa6yqBg.jpeg
  - 1*YAvz5RgxqOMHOZ2aDjn6Jg.png
  - 1*UdfW29NBTyEGP4h42xUs0g.png
  - 1*V5v_nfZmTx8UszIRf2-4yg.png
  - 1*dYTsKX5EBmuokKLX0yOJHA.png
  - 1*XwifxOUPX1IQcJUe0YEyNQ.png
  - 1*T0kJxCSZLTw93G-d363-LA.png
  - featuredImage.png
  - 1*HkmgQKoX0rV1b4e3e-or4w.png
  - 1*uPzCerO1K3atDhvkYbe4LQ.png
  - 1*NE01eYt6NBHuQgnGFyo8mw.png
  - 1*-twHuk2pJunJq_ESOJwQwA.png
tags:
  - rstats
  - nlp
  - tidyverse
  - data_analysis
keywords:
  - data-science
  - nlp
  - tidyverse
  - data-analysis
  - r-language
url: /post/analyzing-tweets-with-r/
---

{{< figure src="featuredImage.jpg" caption="[Source](https://pixabay.com/en/book-dictionary-swedish-german-3101450/)" >}}

## Introduction

NLP(Natural-language processing) is hard, partly because human is hard to understand. We need good tools to help us analyze texts. Even if the texts are eventually fed into a black box model, doing exploratory analysis is very likely to help you get a better model.

I’ve heard great things about a R package _tidytext_ and recently decided to give it a try. The package authors also wrote a book about it and kindly released it online: **[Text Mining with R: A guide to text analysis within the tidy data framework, using the tidytext package and other tidy tools](https://www.tidytextmining.com/)**.

As the name suggests, _tidytext_ aims to provide text processing capability in the [\*tidyverse](https://www.tidyverse.org/)* ecosystem. I was more familiar with *data.table* and its way of data manipulation, and the way *tidyverse* handles data had always seemed tedious to me. But after working through the book, I’ve found the syntax of *tidyverse\* very elegant and intuitive. I love it! All you need is some good examples to help you learn the ropes.

Chapter 7 of the book provides a case study comparing tweet archives of the two authors. Since twitter only allows downloading the user’s own archive, it is hard for a reader without friends (i.e. me) to follow. So I found a way to download tweets of public figures and I’d like to share with you how to do it. This post also presents an example comparing tweets from Donald Trump and Barack Obama. The work flow is exactly the same as in the book.

_Warning: The content of this post may seem very elementary to professionals._

### R tips

- Use [Microsoft R Open](https://mran.microsoft.com/open) if possible. It comes with the multi-threaded math library (MKL) and _checkpoint_ package.

- But don’t hesitate to switch to regular R if you run into trouble. Microsoft R Open can have some bizarre problems in [my personal experiences](https://medium.com/@ceshine/cxx11-is-not-defined-problem-in-mro-3-4-e51f1d27da15). Don’t waste too much time on fixing them. Switching to regular R often solves the problem.

- Install regular R via CRAN ([Instructions for Ubuntu](https://cran.r-project.org/bin/linux/ubuntu/README.html)). [Install _checkpoint_ to ensure reproducibility](https://mran.microsoft.com/documents/rro/reproducibility) (it is not a Microsoft R Open exclusive.)

- Use [RStudio](https://www.rstudio.com/) and make good use of its _Console_ window. Some people hold strong feelings against R because of some of its uniqueness comparing to other major programming languages. In fact a lot of the confusion can be resolved with simple queries in the _Console_. Not sure whether the vector index starts from zero or one? `c(1,2,3)[1]` tells you it’s one. Querying `1:10` tells you the result includes 10 (unlike Python).

{{< figure src="1*YAvz5RgxqOMHOZ2aDjn6Jg.png" >}}

## Getting the data and distribution of tweets

First of all, follow the instruction of this article to obtain your own API key and access token, and install `twitteR` package: **[Accessing Data from Twitter API using R (part1)](https://medium.com/@GalarnykMichael/accessing-data-from-twitter-api-using-r-part1-b387a1c7d3e)**.

You need these four variables:

```R
consumer_key <- "FILL HERE"
consumer_secret <- "FILL HERE"
access_token <- "FILL HERE"
access_secret <- "FILL HERE"
```

The main access point for this post is `userTimeline`. It downloads at most 3200 recent tweets of a public twitter user. The default `includeRts=FALSE` parameter seems to remove a lot of false positives, so we’ll instead do it manually later.

```R
setup_twitter_oauth(
    consumer_key, consumer_secret, access_token, access_secret)

trump <- userTimeline("realDonaldTrump", n=3200, includeRts=T)
obama <- userTimeline("BarackObama", n=3200, includeRts=T)
president.obama <- userTimeline("POTUS44", n=3200, includeRts=T)
```

Now we have tweets from @realDonaldTrump, @BarackObama and @POTUS44 as _List_ objects. We’ll now convert them to data frames:

```R
df.trump <- twListToDF(trump)
df.obama <- twListToDF(obama)
df.president.obama <- twListToDF(president.obama)
```

{{< figure src="1*UdfW29NBTyEGP4h42xUs0g.png" >}}

(The `favorited`, `retweeted` columns are specific to the owner of the access token.) We’re going to only keep columns `text`, `favoriteCount`, `screenName`, `created `and `retweetCount`, and filter out those rows with `isRetweet=TRUE.` (`statusSource` might be [of interest in some application](https://github.com/arm5077/trump-twitter-classify)s.)

```R
tweets <- bind_rows(
  df.trump %>% filter(isRetweet==F) %>%
    select(
      text, screenName, created, retweetCount, favoriteCount),
  df.obama %>% filter(isRetweet==F) %>%
    select(
      text, screenName, created, retweetCount, favoriteCount),
  df.president.obama %>% filter(isRetweet==F) %>%
    select(
      text, screenName, created, retweetCount, favoriteCount))
```

Now we plot the time distribution of the tweets:

```R
ggplot(tweets, aes(x = created, fill = screenName)) +
  geom_histogram(
    position = "identity", bins = 50, show.legend = FALSE) +
  facet_wrap(~screenName, ncol = 1, scales = "free_y") +
  ggtitle("Tweet Activity (Adaptive y-axis)")
```

{{< figure src="1*V5v_nfZmTx8UszIRf2-4yg.png" >}}

You could remove `scales = "free_y"` to have compare the absolute amount of tweets instead of relative:

{{< figure src="1*dYTsKX5EBmuokKLX0yOJHA.png" >}}

(The lack of activity of @realDonaldTrump is from the 3200-tweet restriction) We can see that as a president, Donald Trump tweets a lot more than Barack Obama did.

## Word frequencies

From this point we’ll enter the world of tidyverse:

```R
replace_reg <- "http[s]?://[A-Za-z\\d/\\.]+|&amp;|&lt;|&gt;"
unnest_reg  <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"
tidy_tweets <- tweets %>%
  filter(!str_detect(text, "^RT")) %>%
  mutate(text = str_replace_all(text, replace_reg, "")) %>%
  mutate(id = row_number()) %>%
  unnest_tokens(
    word, text, token = "regex", pattern = unnest_reg) %>%
  filter(!word %in% stop_words$word, str_detect(word, "[a-z]"))
```

(It’s worth noting that the pattern used in `unnest_tokens` are for matching **separators**, not tokens.) Now we have the data in one-token-per-row tidy text format. We can use it to do some counting:

```R
frequency <- tidy_tweets %>%
  group_by(screenName) %>%
  count(word, sort = TRUE) %>%
  left_join(tidy_tweets %>%
              group_by(screenName) %>%
              summarise(total = n())) %>%
  mutate(freq = n/total)

frequency.spread <- frequency %>%
  select(screenName, word, freq) %>%
  spread(screenName, freq) %>%
  arrange(desc(BarackObama), desc(realDonaldTrump))

ggplot(frequency.spread, aes(BarackObama, realDonaldTrump)) +
  geom_jitter(
    alpha = 0.1, size = 2.5, width = 0.15, height = 0.15) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 0) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  geom_abline(color = "red") + theme_bw()
```

{{< figure src="1*XwifxOUPX1IQcJUe0YEyNQ.png" caption="Comparison of Word Frequencies" >}}

(Because of the jitters, the text labels sometimes are far away from its corresponding data point. So far I don’t have a solution for this problem.) One observation from the above plot is the more frequent use of “republicans” , “republican”, and “democrats” by Trump.

## Comparing word usage

```R
word_ratios <- tidy_tweets %>%
  filter(screenName != "POTUS44") %>%
  filter(!str_detect(word, "^@")) %>%
  count(word, screenName) %>%
  filter(sum(n) >= 10) %>%
  ungroup() %>%
  spread(screenName, n, fill = 0) %>%
  mutate_if(is.numeric, funs((. + 1) / sum(. + 1))) %>%
  mutate(logratio = log(realDonaldTrump / BarackObama)) %>%
  arrange(desc(logratio))

word_ratios %>%
  group_by(logratio < 0) %>%
  top_n(15, abs(logratio)) %>%
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio, fill = logratio < 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("log odds ratio (realDonaldTrump/BarackObama)") +
  scale_fill_discrete(name = "", labels = c("realDonaldTrump", "BarackObama"))
```

{{< figure src="1*T0kJxCSZLTw93G-d363-LA.png" caption="Comparison of Word Frequencies" >}}

Readers can find the dramatically different characteristics of the two presidents from the plot. But to be fair, some of the words from Obama are hashtags that are unlikely to be re-used later. Let’s see what will come up when we remove the hashtags:

{{< figure src="1*vh8bQQL7p_g31a5Tq3HjvQ.png" caption="Most Distinctive Words (excluding hastags)" >}}

## Changes in word use

This part is more involved, so I’ll skip the source code. The idea is to fit a glm model to predict the word frequency with the relative point of time. If the coefficient of the point of time is very unlikely to be zero (low p value), we say that the frequency of this word has changed over time. We plot the words with lowest p values for each president below:

{{< figure src="1*HkmgQKoX0rV1b4e3e-or4w.png" caption="Donald Trump" >}}

{{< figure src="1*uPzCerO1K3atDhvkYbe4LQ.png" caption="Barack Obama" >}}

For Barack Obama, only tweets from his presidency has been included because of the sparsity of the tweets in 2017 and beyond. Please check the book or the R Markdown link at the end of the post for source code and more information.

## Favorites and retweets

We can count all the retweets and favorites and count which words are more likely to appear. Note it’s important to count each word only once in every tweet. We achieve this by grouping by _(id, word, screenName)_ and _summarise_ with a *first *function:

```R
totals <- tweets %>%
  group_by(screenName) %>%
  summarise(total_rts = sum(retweetCount))

word_by_rts <- tidy_tweets %>%
  group_by(id, word, screenName) %>%
  summarise(rts = first(retweetCount)) %>%
  group_by(screenName, word) %>%
  summarise(retweetCount = median(rts), uses = n()) %>%
  left_join(totals) %>%
  filter(retweetCount != 0) %>%
  ungroup()

word_by_rts %>%
  filter(uses >= 5) %>%
  group_by(screenName) %>%
  top_n(10, retweetCount) %>%
  arrange(retweetCount) %>%
  ungroup() %>%
  mutate(
    word = reorder(
      paste(word, screenName, sep = "__"),
      retweetCount)) %>%
  ungroup() %>%
  ggplot(aes(word, retweetCount, fill = screenName)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ screenName, scales = "free_y", ncol = 1) +
  scale_x_discrete(labels = function(x) gsub("__.+$","", x)) +
  coord_flip() +
  labs(x = NULL,
       y = "Median
       # of retweetCount for tweets containing each word")
```

{{< figure src="1*NE01eYt6NBHuQgnGFyo8mw.png" caption="Words that are most likely to be in a retweet" >}}

Use the same code as above, but replace `retweetCount` with `favoriteCount`:

{{< figure src="1*-twHuk2pJunJq_ESOJwQwA.png" caption="Words that are most likely to be in a favorite" >}}

There’s a interesting change of pattern between Trump’s retweets and favorites. It seems there are some tweets people would rather retweet than favorite, and vice versa.

## The End

Thanks for reading! Please [support the author of the book](http://shop.oreilly.com/product/0636920067153.do?cmp=af-strata-books-video-product_cj_0636920067153_4428796) (I have no affiliation with them) if you like the `tidytext` package and the book.

You can find the source code of this post on RPubs: **[RPubs - Trump & Obama Tweets Analysis](http://rpubs.com/ceshine/tweet_analysis)**.
