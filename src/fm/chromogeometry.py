from enum import Enum
import numpy as np


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


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


def matrix(x: float, y: float, color: Color) -> np.ndarray:
    match color:
        case Color.RED:
            return np.array([
                [  x,  y ],
                [  y,  x ],
            ])
        case Color.GREEN:
            return np.array([
                [  x,  0 ],
                [  0,  y ],
            ])
        case Color.BLUE:
            return np.array([
                [  x,  y ],
                [ -y,  x ],
            ])


def x_component(z: np.ndarray, color: Color) -> float:
    match color:
        case Color.RED:
            return z[0, 0]
        case Color.GREEN:
            return z[0, 0]
        case Color.BLUE:
            return z[0, 0]
    

def y_component(z: np.ndarray, color: Color) -> float:
    match color:
        case Color.RED:
            return z[0, 1]
        case Color.GREEN:
            return z[1, 1]    
        case Color.BLUE:
            return z[0, 1]


def quadrance(z: np.ndarray) -> float:
    return z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]


def conjugate(z: np.ndarray) -> np.ndarray:
    z[0, 1] = -z[0, 1]
    z[1, 0] = -z[1, 0]

    return z
