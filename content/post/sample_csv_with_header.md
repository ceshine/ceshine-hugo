+++
Categories = ["Data Engineering"]
Description = ""
Tags = ["data_eng"]
date = "2015-07-31T14:09:12+08:00"
title = "Random Sampling Data with Header"

+++

[I've mention](/tip/sampling_data) a handy script call **sample**, which can randomly sample row/record-based data with given probability. One major problem with this script is that it doesn't consider data with a header row specifying field names. It samples the head row like every other row. It's not the end of the world though; two lines of **head** and **cat** commands can easily fix that. But it has become more and more annoying to do this every time.

So I've modified the original script to take headers into account. The code is on [Github Gist](https://gist.github.com/ceshine/ced4787bd41555b729de).

Simply add *--header* in the command whenever you need to ensure the header is included in the output:

```bash
sample -r 10% --header < input.csv > output.csv
```