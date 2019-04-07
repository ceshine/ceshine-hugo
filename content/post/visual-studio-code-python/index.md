---
slug: use-visual-studio-code-to-develop-python-programs
date: 2018-09-25T00:50:36.623Z
title: "Use Visual Studio Code To Develop Python Programs"
description: "From a perspective of a long time Emacs user"
images:
  - featuredImage.jpeg
tags:
  - python
  - vscode
  - tool
keywords:
  - python
  - visual-studio-code
  - vscode
  - ide
url: /post/use-visual-studio-code-to-develop-python-programs/
---

{{< figure src="featuredImage.jpeg" caption="[Photo Credit](https://pixabay.com/en/drop-of-water-dew-close-up-nature-3671613/)" >}}

Joel Grus offered his critique of Jupyter Notebook in a recent talk. I think most of his points are valid and recommend you to [read the slides](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g3cbe089527_0_12) or [watch the talk](https://www.youtube.com/watch?v=7jiPeIFXb6U). However, one thing that caught my attention is Mr. Grus’s Python IDE(Visual Studio Code). It looks so good that I decided to give it a try, which led to this blog post.
[**I Don't Like Notebooks - Joel Grus - #JupyterCon 2018**
*I Don't Like Notebooks hi, I'm Joel, and I don't like notebooks Joel Grus (@joelgrus) #JupyterCon 2018*docs.google.com](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g362da58057_0_1)

*Some disclaimers first: Which IDE to use is a very personal choice. I don’t claim that Visual Studio Code is the best Python IDE that everyone should use. This is just my opinionated review, and what works best for my use case might not work best for yours.*

# Comparing to Other Modern IDEs

I tried using Atom and PyCharm to write Python code before. Python support in Atom wasn’t powerful nor easy to use enough for me. And PyCharm is too heavy-weight and a simple format-on-save [requires some dancing around](https://stackoverflow.com/questions/44483748/pycharm-pep8-on-save) to be done. I think Visual Studio Code has found a good middle ground: it’s lightweight enough and yet still powerful enough.

## Comparing to Emacs

For years Emacs has been served me well. And I’ll continue to use it when I have to work in command line. The Python support of Emacs is really mature. Here are some packages I put in my dotfiles and sync to every computer I use:

* elpy

* flycheck(pylint)

* anaconda-mode

* py-autopep8

* jedi

* magit

IMO Visual Studio Code and its extensions are on par with these packages, if not better. Some Emacs folks [shared their experiences with VS Code](https://www.reddit.com/r/emacs/comments/8h1cxa/any_long_time_emacs_users_tried_vscode/) and pointed out the license issue. I’m not sure I should care or not, but other than that the comments were more or less positive.

Some VS Code Highlights:

* Onboarding: VS Code is very easy to learn and use. Extensions come with sensible default configurations.

* Terminal: the embedded terminal is pretty close to the GNOME terminal. While the one in Emacs is almost unusable without some serious customization.

* Git integration: some say *magit *in Emacs is really good, but usually I’d rather directly use Git in the command line, and use *gitk* when I want a graphical interface. Built-in Git function in VS Code is good enough, and you can even install extensions like *gitlens* to do more.

* Built-in support for various linters, including mypy. (There are [emacs-flycheck-mypy](https://github.com/lbolla/emacs-flycheck-mypy) and [mypy-mode](https://github.com/SerialDev/mypy-mode) Emacs packages.)

* Command palette: should be straight-forward to use for Emacs users (*M-x command*).

* Nice built-in debugger interface.

## Bootstrap Your Visual Studio Code Environment

I’ve written a short markdown document and published it on Github Gist. It’s still a work in progress and I’ll keep updating it with things I’ve found useful. (I’ve only started using VS Code for a few days.)

{{< gist ceshine 0054390380b9e341c66d36a8b5e2a48b >}}