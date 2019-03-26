+++
Categories = [ "Python", "Statistics" ]
Description = "confidence interval, correlation coefficient, and linear regression"
Tags = []
date = "2014-02-25T17:54:38+08:00"
title = "Shortcuts for some common statistical functions"
+++

Here are some useful functions when performing statistical analysis:

Confidence Interval
-------
```python
from scipy import stats
from numpy import mean, var
from math import sqrt

sample = [1, 2, 3, 4, 5]

#95% confidence interval
R = stats.t.interval(0.95, len(sample)-1, loc=mean(sample),
                     scale=sqrt(var(sample)/len(sample)))
>>> R
(1.2440219338298311, 4.7559780661701687)
```

[SciPy documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)

Correlation Coefficient
--------
```python
from numpy import corrcoef
x = [1, 2, 3, 4, 100]
y = [6, 7, 8, 9, 10]

r = corrcoef(x, y)  

>>> r
array([[ 1., 0.72499943],
       [ 0.72499943,  1.]])

```

[SciPy documentation](http://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html)

Linear Regression
--------
```python
from scipy import stats
x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

>>> slope, intercept
(1.0, 5.0)
```

[SciPy documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)
