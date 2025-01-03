from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from fm import constants
from fm.generator import Generator


class Mode(Enum):
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3


class Transformer:
    def __init__(self) -> None:
        self.generators = [ Generator() for _ in range(3) ]

        self.mode = Mode.DOUBLE

        self.generators[0].active = self.mode.value >= Mode.SINGLE.value
        self.generators[1].active = self.mode.value >= Mode.DOUBLE.value
        self.generators[2].active = self.mode.value >= Mode.TRIPLE.value

        self.generators[0].set_coefficients(constants.COEFFICIENTS[0])
        self.generators[1].set_coefficients(constants.COEFFICIENTS[1])
        self.generators[2].set_coefficients(constants.COEFFICIENTS[2])

        self.generators[0].set_exponents(constants.EXPONENTS[0])
        self.generators[1].set_exponents(constants.EXPONENTS[1])
        self.generators[2].set_exponents(constants.EXPONENTS[2])

        self.angle = 2.0 * np.pi / constants.FRAME_COUNT

        self.axes = np.array(constants.AXES)
        self.axes /= np.linalg.norm(self.axes, axis=1, keepdims=True)

        self.set_mode_weights()

        self.rotations = [
            Rotation.from_rotvec(self.angle * axis).as_matrix() 
            for axis in self.axes 
        ]


    def set_mode_weights(self) -> None:
        match self.mode:
            case Mode.SINGLE:
                self.weights = np.array([
                    [0.847, 0.659, 0.753],
                ])
            case Mode.DOUBLE:
                self.weights = np.array([
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                ])
            case Mode.TRIPLE:
                self.weights = np.array([
                    [ 1.0, 0.0, 0.0 ],
                    [ 0.0, 1.0, 0.0 ],
                    [ 0.0, 0.0, 1.0 ],
                ])


    def step(self) -> np.ndarray:
        for index, generator in enumerate(self.generators):
            if generator.active:
                generator.print_terms()
                generator.calculate()

                generator.coefficients = self.rotations[index] @ generator.coefficients

        return self.get_pixel_array()


    def get_pixel_array(self) -> np.ndarray:
        histogram = np.zeros(
            (constants.FRAME_SIZE, constants.FRAME_SIZE, 3), 
            dtype=np.float64
        )

        for generator, weight in zip(self.generators, self.weights):
            if generator.active:
                histogram += weight * generator.histogram[:, :, np.newaxis]
        
        return (histogram * 255).astype(np.uint8)
