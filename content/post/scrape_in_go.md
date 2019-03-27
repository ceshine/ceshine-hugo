+++
Categories = ["golang"]
Description = ""
Tags = ["golang"]
date = "2015-08-29T15:26:20+08:00"
title = "First Step of Web Scraping in Go"
+++

An appropriate amount of web scraping is often required for web-related data science projects. Python has a well-known scraping framework called [Scrapy](http://www.wikiwand.com/en/Scrapy) which aims to accommodate all kinds of possible scenarios. For those who want more control over the process and don't mind getting their hands dirty, [GRequests](https://github.com/kennethreitz/grequests)(or the good old [Requests](https://github.com/kennethreitz/requests)) combined with [BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/) can also be a solid solution.

However, multi-threading in Python can cause a lot of pain in the neck. And Scrapy depends on [Twisted](https://twistedmatrix.com/trac/), which is not yet Python3-ready, and there is no clear roadmap on when the project will finish migrating to Python 3.x. These constraints made me started finding other faster, and more robust alternatives.

Go programming language is known for its built-in concurrency support, and being a lower-level language with some higher-level features. So it came to me naturally that I should try to scrape with Go(Admittedly, This process of selection was biased because I already have some experiences in Go).

After some search, I've found a few repositories on Github doing web scraping in Go, but most of them are months or years old and aren't maintained anymore. Finally I decided that [web-scraping-example](https://github.com/kyokomi/web-scraping-example) is a good place to start. It is based on [goquery](https://github.com/PuerkitoBio/goquery), which brings jQuery-like syntax to Go. My preliminary experience shows that goquery is very powerful and definitely has the potentials to match BeautifulSoup. The *web-scraping-example* program was originally designed to scrape a single page and download images into the designated folder, but with a few modifications it can suit a wide range of scraping tasks.

I've made some improvements on the program. Firstly the program can now scrape through consecutive, numbered pages. And I boosted the concurrency using sync.WaitGroup [(reference)](http://blog.golang.org/pipelines). Below I attached the part of code that implement this feature:

```go
var wg sync.WaitGroup
done := make(chan struct{}, 1)
urls := make(chan string, 5)
var url string
for _, page := range configFile.PageSettings {
  go page.GetImagePaths(configFile.Keyword, urls, done)
  wg.Add(1)
  go func(wg *sync.WaitGroup) {
    for {
      select {
      case url = <-urls:
        wg.Add(1)
        go func(writeDir, url string) {
          writeImage(writeDir, url)
          wg.Done()
        }(writeDir, url)
      case <-done:
        wg.Done()
        return
      }
    }
  }(&wg)
}
wg.Wait()
```

**GetImagePaths** will push the urls of the images to the channel **urls**, and push an empty struct *struc{}{}* to channel **done** when done. I could call *wg.Done()* inside GetImagePaths and avoid using channel **done**, but I think it'll be clearer if I keep all the operations to *wg* in one place.

The results are already quite satisfactory to me. Though it's very hard to generalize the program by considering more scraping scenarios, one can start from this small example to quickly create a specialized scraping program which fits their needs.

(The complete code is at https://github.com/ceshine/web-scraping-example. I haven't updated README there yet, but I plan to.)
