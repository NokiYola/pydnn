import numpy as np


def labelsToMat(Y_Labels, outputSize):
    nbEx = len(Y_Labels)
    Y = np.array(np.zeros((nbEx, outputSize), dtype=float))
    Y[range(nbEx), Y_Labels] = 1

    return Y
