import math

import numpy as np

from fm import constants
from fm.generator import Generator


class Transformer:
    def __init__(self) -> None:
        self.angle = 0.0
        self.axis = np.array([0, 1, 0])
        
        self.generators = [Generator() for _ in range(3)]

        self.generators[0].active = True
        self.generators[1].active = False
        self.generators[2].active = False


    def step(self) -> None:
        if self.generators[0].active:
            self.generators[0].find_border()
            self.generators[0].calculate()

        if self.generators[1].active:
            self.generators[1].find_border()
            self.generators[1].calculate()

        if self.generators[2].active:
            self.generators[2].find_border()
            self.generators[2].calculate()

        self.rotate(2.0 * np.pi / constants.FRAME_COUNT)


    def rotate(self, increment: float) -> None:
        self.angle += increment

        self.calculate_rotation_matrix()

        if self.generators[0].active:
            self.generators[0].coefficients = self.rotation_matrix @ self.generators[0].coefficients

        if self.generators[1].active:
            self.generators[1].coefficients = self.rotation_matrix @ self.generators[1].coefficients

        if self.generators[2].active:
            self.generators[2].coefficients = self.rotation_matrix @ self.generators[2].coefficients


    def calculate_rotation_matrix(self) -> None:
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        t = 1 - c

        x, y, z = self.axis

        self.rotation_matrix = np.array([
            [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c    ]
        ])


    def get_pixel_array(self) -> np.ndarray:
        pixel_array = np.zeros((3, constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)

        if self.generators[0].active:
            histogram_pixel0 = (self.generators[0].histogram * 255).astype(np.uint8)
            pixel_array += self.generators[0].weight * histogram_pixel0
        
        if self.generators[1].active:
            histogram_pixel1 = (self.generators[1].histogram * 255).astype(np.uint8)
            pixel_array += self.generators[1].weight * histogram_pixel1

        if self.generators[2].active:
            histogram_pixel2 = (self.generators[2].histogram * 255).astype(np.uint8)
            pixel_array += self.generators[2].weight * histogram_pixel2

        pixel_array = pixel_array.astype(np.uint8)
        pixel_array = pixel_array.transpose(1, 2, 0)

        return pixel_array
