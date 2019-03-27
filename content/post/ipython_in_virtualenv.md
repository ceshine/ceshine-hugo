+++
Categories = [ "Python" ]
Description = "This should help you avoid a common pitfall you'd encounter trying to use ipython notebooks in virtualenv"
Tags = ["python"]
date = "2014-04-29T13:54:38+08:00"
title = "Tip for using iPython Notebooks in virtualenv"
+++

When trying to install ipython and dependencies of its notebook feature via pip, I was stuck. Even I'd already installed pyzmq, I still got this message:

```bash
ImportError: IPython.zmq requires pyzmq
```

It was quite frustrating, until I found this [post on StackOverflow](http://stackoverflow.com/questions/17992077/setup-ipython-notebook-in-virtualenv).

So it turns out this can be solved by just install pyzmq using an extra parameter:

```bash
pip install pyzmq --install-option="--zmq=bundled"
```
