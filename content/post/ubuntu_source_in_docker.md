+++
Categories = [ "Docker" ]
Description = ""
Tags = ["docker"]
date = "2015-07-07T17:42:55+10:00"
title = "Change Sources of Ubuntu in a Docker image"

+++

The official docker images of Ubuntu use _archive.ubuntu.com_ as the default package source.

Because my Internet connection is metered, I'd like to change it to the free mirror server my ISP provides.

And this command does the trick:

``` bash
sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirror.internode.on.net\/pub\/ubuntu/g' /etc/apt/sources.list
```

Change **http:\/\/mirror.internode.on.net\/pub\/ubuntu** to the url of whatever mirror you prefer.

And add **RUN** to the start of the line and put it into the Dockerfile.
