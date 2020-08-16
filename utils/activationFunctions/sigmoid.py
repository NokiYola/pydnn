import numpy as np


def fun(z):
    return 1 / (1 + np.exp(-z))


def der(z):
    return np.multiply(fun(z), (1 - fun(z)))
