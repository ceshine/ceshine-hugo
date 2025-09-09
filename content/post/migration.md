---
categories: ["Docker"]
description: "A guide on how to migrate a blog from Pelican to Hugo, and how to deploy it with Docker and Docker-Compose."
tags: ["docker"]
date: "2015-07-28T17:00:36+10:00"
title: "Migrated the Blog from Pelican to Hugo"
---

I've been using [pelican](http://blog.getpelican.com/) to build _blog.ceshine.net_ for about two years, and as you can see, I've not been very productive. Part of the reasons is that I found I spent more time tuning the code rather than actually writing stuffs.

Recently Go-based [Hugo](http://gohugo.io/) caught my attention. Go can easily compile multi-platform binary executables, which makes deployment much easier. Hugo also provide a decent built-in web server whose performance is good enough for some small-scale production use. So after some experimenting, I decided to replace the old pelican site with Hugo.

The deployment will be even more simpler if you use Docker and Docker-Compose. First put the site under git version control, in my case, I put it on https://github.com/ceshine/ceshine-hugo.git. We're gonna need two images, **content** and **hugo**:

The dockerfile for **content**:

```bash
FROM ubuntu:14.04

RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates git-core
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git clone --recursive https://github.com/ceshine/ceshine-hugo.git /src/blog

VOLUME ["/src/blog"]
WORKDIR /src/blog

ENTRYPOINT ["git"]
CMD ["pull"]
```

The dockerfile for **hugo**:

```bash
FROM ubuntu:14.04

RUN apt-get install -y curl
# Fetch and install the hugo binary files
RUN curl -L -o /tmp/hugo.tar.gz https://github.com/spf13/hugo/releases/download/v0.14/hugo_0.14_linux_amd64.tar.gz
RUN tar zxvf /tmp/hugo.tar.gz -C /tmp  && mv /tmp/hugo_0.14_linux_amd64/hugo_0.14_linux_amd64 /usr/local/bin/hugo

# VOLUME ["/var/www/blog"]

RUN apt-get install -y python-pygments
ENTRYPOINT ["hugo"]
#CMD ["-w", "-s", "/src/blog", "-d", "/var/www/blog"]
CMD ["server", "-w", "-s", "/src/blog", "--bind=0.0.0.0", "--appendPort=false", "-v", "-b", "https://hugo.ceshine.net", "--disableLiveReload"]
```

For Docker-Compose, docker-compose.yml:

```yaml
content:
  build: dockerfiles/content/

web:
  build: dockerfiles/hugo/
  ports:
    - 80:1313
  volumes_from:
    - content
  restart: on-failure
```

To start the server on port 80:

```bash
docker-compose up -d
```

To fetch the new content off the git remote and automatically update the site:

```bash
docker-compose up content
```
