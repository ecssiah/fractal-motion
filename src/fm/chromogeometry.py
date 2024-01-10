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


def half_trace(z: np.ndarray) -> float:
    trace = z[0, 0] + z[1, 1]

    return (1/2) * trace


def quadrance(z: np.ndarray) -> float:
    return z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]


def adjugate(z: np.ndarray) -> np.ndarray:
    z[0, 0], z[1, 1] = z[1, 1], z[0, 0]
    z[1, 0] = -z[1, 0]
    z[0, 1] = -z[0, 1]

    return z
