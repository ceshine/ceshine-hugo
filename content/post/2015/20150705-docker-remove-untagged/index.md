---
date: "2015-07-05T18:09:20+10:00"
draft: false
title: "Docker: Remove All Untagged Images"
description: "Learn how to remove all untagged Docker images and clean up old containers."
tags: ["docker"]
categories: [
  "Docker"
]
---
By courtesy of this [post](http://jimhoskins.com/2013/07/27/remove-untagged-docker-images.html), its comment section and this [thread](http://stackoverflow.com/questions/17236796/how-to-remove-old-docker-containers):

## Clean up old containers
``` bash
docker ps -a | grep 'Exited' | awk '{print $1}' | xargs --no-run-if-empty docker rm
```

## Remove All Untagged Images
``` bash
docker rmi $(docker images -q --filter "dangling=true")
```
