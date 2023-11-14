import numpy as np
from numba import njit


@njit
def mean_(n: float):
    return sum(n) / len(n)


@njit
def quantile_(n: list):
    return np.quantile(n, .75)


