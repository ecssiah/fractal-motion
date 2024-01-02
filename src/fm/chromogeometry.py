import numpy as np

IDENTITY = np.array([
    [ 1,  0],
    [ 0,  1],
])

RED = np.array([
    [ 0,  1],
    [ 1,  0],
])

GREEN = np.array([
    [ 1,  0],
    [ 0, -1],
])

BLUE = np.array([
    [ 0,  1],
    [-1,  0],
])


def conjugate(z):
    z[0, 1] *= -1
    z[1, 0] *= -1

    return z