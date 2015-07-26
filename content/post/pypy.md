+++
Categories = [ "Python" ]
Description = "Using PyPy could help a lot in machine learning challenges/competitions"
Tags = []
date = "2014-11-29T13:54:38+08:00"
title = "The Power of PyPy"
+++

[PyPy](http://pypy.org/) is an alternative Python implementation which emphasize on speed and memory usage. I didn't take it seriously until I wrote a Python script for a kaggle competition that requires hours to run. I read someone on the kaggle forum suggesting everyone to give PyPy a try. I did. And it worked like a magic. A 2 to 5 times speed boost can be achieved just by substituting **python** with **pypy** when you run a python script. Don't have a accurate number for that, but it was significantly faster. This is critical because now you have more time to try different models and hence get a better score in the competition.

Of course, there are reasons that PyPy hasn't replaced CPython, the official implementation. One of them is the compatibility with external libraries. Particularly for machine learning pratitioners, [Numpy](http://pypy.org/compat.html) support is not complete; you have to install pypy's own fork of numpy. So PyPy is most powerful for scripts that use only the core function of Python, e.g. a simple implementation of Logistic SGD.
