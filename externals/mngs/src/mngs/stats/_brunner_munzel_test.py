#!/usr/bin/env python3

import numpy as np
from scipy import stats


def brunner_munzel_test(x1, x2, distribution="t"):
    """Calculate Brunner-Munzel-test scores.
    Parameters:
      x1, x2: array_like
        Numeric data values from sample 1, 2.
    Returns:
      w:
        Calculated test statistic.
      p_value:
        Two-tailed p-value of test.
      dof:
        Degree of freedom.
      p:
        "P(x1 < x2) + 0.5 P(x1 = x2)" estimates.
    References:
      * https://oku.edu.mie-u.ac.jp/~okumura/stat/brunner-munzel.html
    Example:
      When sample number N is small, distribution='t' is recommended.
      d1 = np.array([1,2,1,1,1,1,1,1,1,1,2,4,1,1])
      d2 = np.array([3,3,4,3,1,2,3,1,1,5,4])
      print(bmtest(d1, d2, distribution='t'))
      print(bmtest(d1, d2, distribution='normal'))
      When sample number N is large, distribution='normal' is recommended; however,
      't' and 'normal' yield almost the same result.
      d1 = np.random.rand(1000)*100
      d2 = np.random.rand(10000)*110
      print(bmtest(d1, d2, distribution='t'))
      print(bmtest(d1, d2, distribution='normal'))
    """

    n1, n2 = len(x1), len(x2)
    R = stats.rankdata(list(x1) + list(x2))
    R1, R2 = R[:n1], R[n1:]
    r1_mean, r2_mean = np.mean(R1), np.mean(R2)
    Ri1, Ri2 = stats.rankdata(x1), stats.rankdata(x2)
    var1 = np.var([r - ri for r, ri in zip(R1, Ri1)], ddof=1)
    var2 = np.var([r - ri for r, ri in zip(R2, Ri2)], ddof=1)
    w = ((n1 * n2) * (r2_mean - r1_mean)) / ((n1 + n2) * np.sqrt(n1 * var1 + n2 * var2))
    if distribution == "t":
        dof = (n1 * var1 + n2 * var2) ** 2 / (
            (n1 * var1) ** 2 / (n1 - 1) + (n2 * var2) ** 2 / (n2 - 1)
        )
        c = stats.t.cdf(abs(w), dof) if not np.isinf(w) else 0.0
    if distribution == "normal":
        dof = np.nan
        c = stats.norm.cdf(abs(w)) if not np.isinf(w) else 0.0
    p_value = min(c, 1.0 - c) * 2.0
    p = (r2_mean - r1_mean) / (n1 + n2) + 0.5
    return (w, p_value, dof, p)
