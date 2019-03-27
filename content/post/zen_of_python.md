+++
Categories = [ "Python" ]
Description = ""
Tags = ["python"]
date = "2013-10-07T13:54:38+08:00"
title = "Dicussing the zen of python"
+++

Every newbie for python should have already heard of or read [PEP-8 a.k.a. THE style guide for python code](http://www.python.org/dev/peps/pep-0008/), which hopefully I can cover in one of the next few posts. However, there's one more important ground for you to cover before you can become a professional. It's [The Zen of Python](http://www.python.org/dev/peps/pep-0020/). You can read it right inside the python interpreter:

```python
import this
```

The Zen of Python
------
* Beautiful is better than ugly.
* Explicit is better than implicit.
* Simple is better than complex.
* Complex is better than complicated.
* Flat is better than nested.
* Sparse is better than dense.
* Readability counts.
* Special cases aren't special enough to break the rules.
* Although practicality beats purity.
* Errors should never pass silently.
* Unless explicitly silenced.
* In the face of ambiguity, refuse the temptation to guess.
* There should be one-- and preferably only one --obvious way to do it.
* Although that way may not be obvious at first unless you're Dutch.
* Now is better than never.
* Although never is often better than *right* now.
* If the implementation is hard to explain, it's a bad idea.
* If the implementation is easy to explain, it may be a good idea.
* Namespaces are one honking great idea -- let's do more of those!

Interpretations and Confusions
------
The Zen of Python describes the philosophy the creators of Python hold when designing Python. However, official interpretations for these 19 aphorisms do not exist except PEP-8. This somewhat creates confusions for Python learners to understand them and to apply them, I myself included.

The vaguest one should be **readability counts**. I've seen a few people preferring packing a large amount of operations into one or two lines of codes to spreading them into several lines and maybe a little nested structures. They claimed one-or-two-liners are more simple, elegant, and clean. What they implied is that this way the codes would be more readable. I highly disagreed. IMHO, one-or-two-liners are only for those programmers to show the world how intelligent or smart they are. It doesn't help others to understand the code more easily.

I hope to write more about this in the future. I only came to realize the importance of the Zen of Python after I had started to work with other Python developers. I used to work solo as the only back-end developer in the team. It's critical for every Python developers in the team to have a correct, or at least coherent understanding of the philosophy behind the language we used to make a living.
