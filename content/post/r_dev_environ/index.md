---
date: 2019-01-03T12:02:54.970Z
Description: "Using Docker and the packrat package"
Categories: ["Machine Learning"]
title: "More Portable, Reproducible R Development Environment"
tags:
  - docker
  - rstats
url: /post/more-portable-reproducible-r-development-environment/
---

{{< figure src="featuredImage.jpeg" caption="[Photo Credit](https://unsplash.com/photos/BzrrAFlc2uk)" >}}

R is awesome. In my opinion it’s the best (free) tool for telling great stories with data. [My first post on Medium](https://medium.com/me/stats/post/e51f1d27da15?source=main_stats_page) was about R. Although what I wrote here mostly involves Python, I still try to get back to R from time to time.

I briefly mentioned my preferred R setup in this previous post “[Analyzing Tweets with R](https://medium.com/the-artificial-impostor/analyzing-tweets-with-r-92ff2ef990c6)” (in “R tips” section), which includes _Microsoft R Open _(_MRO_) and the _checkpoint_ package. Unfortunately, _checkpoint_ doesn’t work well with RStudio, and some weird issues with MRO become more and more annoying to me. Therefore I decided to find a new setup that can work more smoothly and reliably. After some trial and error, here is a configuration that I ended up most satisfied with:

- Use a slightly modified the **[rocker/rstudio](https://github.com/rocker-org/rocker/wiki/Using-the-RStudio-image) Docker image** to provide R base environment and RStudio over the web interface. (This replaces manually installing R via CRAN and RStudio.)

- Use the **[packrat](https://rstudio.github.io/packrat/) package** in every projects to automatically manage package dependencies. (This replaces the _checkpoint_ package)

The above configuration brings the following benefits:

- The web version RStudio works almost the same as the native version (which is an impressive feat!). For some reason the number pad does not work in the RStudio (native) on my Linux Mint system. The web version conveniently solves the problem for me.

- All the good things comes with Docker. Never have to worry about installing R and RStudio on a different machine (a `docker pull` does the trick). Protect your host system against risky operations in your R program. [The list goes on](https://www.quora.com/What-are-the-benefits-using-Docker).

- The web interface allows you to deploy R on a powerful machine, and write code on your laptop. This way you can easily load and manipulate GBs of data into memory while using a low-end laptop.

- The _checkpoint_ package makes sure the versions of the packages used in the project are consistent everywhere by forcing R to install packages from a snapshot of CRAN on a specific day. The _packrat_ package, on the other hand, manages to get the same consistency while at the same time allowing users to choose versions of packages independently (that is, not restricting to the versions available when the snapshot was taken).

- The _packrat_ package gives each project its own package library (similar to what [npm](https://www.npmjs.com/) does by default). This adds nothing new if you create a new Docker container for every new project, but can be handy if you decided to share the same container among projects. However, one drawback of this approach is that you spend more time installing some common packages.

The following sections will give you some more concrete instructions on how to configure this particular R development environment.

# Docker Image

(This section assume you already [installed Docker in your system](https://www.docker.com/get-started), and your operating system is Unix-like.)

The original _rocker/rstudio_ image misses some system packages required by *devtools *and* tidyverse*. The following Dockerfile installs them for you (download and run `docker build -t <image_name> -f Dockerfile .`).

{{< gist ceshine 3b9b4ae613c57a746eb980944a014a7d >}}

To start a new container, run the following line:

```
docker run -d -v (pwd):/home/rstudio/src -e USERID=1000 --name <container_name> -e PASSWORD=<password> -p 8787:8787 rstudio
```

Change the value of `USERID` accordingly. Use the `id` command to find the id of the current user if you are not sure.

Then visit [http://localhost:8787](http://localhost:8787) to access RStudio. Use username _rstudio_ and the password you just set to log in. This command mount your current directory onto `/home/rstudio/src` in the container.

{{< figure src="1*Cp8j8Xc_0ItRmd8N1003Ng.png" link="1*Cp8j8Xc_0ItRmd8N1003Ng.png" title="RStudio in Browser">}}

# Packrat

Create a project in RStudio if you haven’t already. Open _Tools/Project Options_, open the “Packrat” tab, and click on “Use packrat with this project”. RStudio will install _packrat_ for you and initialize it for the current project.

{{< figure src="1*pNiFT_w7osVayg0ps3GWbA.png" width="80%">}}

Now install all the required packages as usual. When you are done. Run `packrat::snapshot()` to tell _packrat_ to write down all the packages you’ve installed and their respective versions.

Inside `packet/` directory, you’ll want to add `init.R`, `packrat.lock`, and `packrat.opts` to your version control system. Everything else is disposable.

{{< figure src="1*_bzUZLDwylM12o5LzhtX5A.png">}}

When you clone the code to a new system, or screw up the library, simply (re-)open the project and run `packrat::restore()`. It tells _packrat_ to (re-)install the packages according to the file `packrat.lock`, so you’ll have the exactly the same library as before.

Not only _packrat_ keeps tracks of packages installed via `install.packages`, those installed via `devtools:install_github` are also included. Here’s what the entry would look like in `packrat.lock`:

{{< figure src="1*rENTITpiy3IhYJU0yfIWtg.png">}}

As you can see, it stores the identifiers of the Github repo, along with a hash signature to ensure integrity. This is tricky because the master branch can be updated frequently. A better way is to lock on a specific commit or tag. (Note that is is a public repo. I’m not sure how private repos work with _packrat_. You’ll have to find out yourself.)

# Additional Resources

- [Packrat: Reproducible package management for R - Walkthrough](https://rstudio.github.io/packrat/walkthrough.html)
- [Packrat: Reproducible package management for R - Rstudio](https://rstudio.github.io/packrat/rstudio.html)

(This post is also published on [Medium](https://medium.com/the-artificial-impostor/more-portable-reproducible-r-development-environment-c3074df7a6a8)).
