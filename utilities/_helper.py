import numpy as np


def myDCT(x):
    n = len(x)
    W = np.cos(2*np.pi * np.outer(np.arange(n), np.arange(n))/n)
    return W @ x


def window_sliding(data, hwin=10):
    l = len(data)
    res = []
    data = np.array(data)
    for i in range(l):
        lb, ub = max(0, i-hwin), min(l-1, i+hwin)
        res.append(data[lb:ub+1].mean())
    return np.array(res)