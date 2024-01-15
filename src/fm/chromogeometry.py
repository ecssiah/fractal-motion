import numpy as np


IDENTITY = np.array([
    [  1,  0 ],
    [  0,  1 ],
])

RED = np.array([
    [  0,  1 ],
    [  1,  0 ],
])

GREEN = np.array([
    [  1,  0 ],
    [  0, -1 ],
])

BLUE = np.array([
    [  0,  1 ],
    [ -1,  0 ],
])


def real(z: np.ndarray) -> float:
    return z[0, 0]


def imag(z: np.ndarray) -> float:
    if z[0, 1] == 0:
        return z[1, 1]
    else:
        return z[0, 1]


def matrix_red(x: float, y: float) -> np.ndarray:
    return np.array([
        [  x,  y ],
        [  y,  x ],
    ])


def matrix_green(x: float, y: float) -> np.ndarray:
    return np.array([
        [  x,  0 ],
        [  0,  y ],
    ])


def matrix_blue(x: float, y: float) -> np.ndarray:
    return np.array([
        [  x,  y ],
        [ -y,  x ],
    ])


def quadrance(z: np.ndarray) -> float:
    return z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]


def conjugate(z: np.ndarray) -> np.ndarray:
    z[0, 1] = -z[0, 1]
    z[1, 0] = -z[1, 0]

    return z
