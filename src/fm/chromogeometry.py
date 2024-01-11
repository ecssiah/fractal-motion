import numpy as np

IDENTITY = np.array([
    [ 1,  0],
    [ 0,  1],
])

BLUE = np.array([
    [ 0,  1],
    [-1,  0],
])

RED = np.array([
    [ 0,  1],
    [ 1,  0],
])

GREEN = np.array([
    [ 1,  0],
    [ 0, -1],
])


def quadrance(z: np.ndarray) -> float:
    return z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]


def conjugate(z: np.ndarray) -> np.ndarray:
    z[0, 1] = -z[0, 1]
    z[1, 0] = -z[1, 0]

    return z
