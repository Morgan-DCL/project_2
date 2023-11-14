from numba import njit
import numpy as np


@njit
def mean_(n: float):
    return sum(n) / len(n)


# @njit
# def quantile_(n: list):
#     return np.quantile(n, .75)

import nltk

