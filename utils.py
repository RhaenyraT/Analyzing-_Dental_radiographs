"""
Some common methods used throughout the rest of the code.
"""

import time
import numpy as np


class Timer(object):
    """
    To time the execution of a piece of code.
    Usage:
        with Timer(dots+"I do someting"): # dots for elegant printing.
    """
    def __init__(self, name=None,  dots=""):
        self.name = name
        self.dots = dots

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            print(self.name)

    def __exit__(self, type, value, traceback):
        print(self.dots + "Elapsed: {}".format( (time.time() - self.tstart)) )
        

def medfilt(x, k):
    """
    Apply a length-k median filter to a 1D array x.
    Bsed on: https://gist.github.com/bhawkins/3535131
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)