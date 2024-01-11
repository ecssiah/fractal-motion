import numpy as np
from scipy.spatial.transform import Rotation

from fm import constants
from fm.generator import Generator


class Transformer:
    def __init__(self) -> None:
        self.axis = np.array([1.0, 0.0, -1.0])
        self.axis /= np.linalg.norm(self.axis)

        self.angle = 2.0 * np.pi / constants.FRAME_COUNT

        self.rotation = Rotation.from_rotvec(self.angle * self.axis).as_matrix()
        
        self.generators = [Generator() for _ in range(3)]

        self.generators[0].active = True
        self.generators[1].active = False
        self.generators[2].active = False

        self.generators[0].set_weight(0.0, 1.0, 1.0)


    def step(self) -> None:
        if self.generators[0].active:
            print(self.generators[0].coefficients)

            self.generators[0].find_border()
            self.generators[0].calculate()

        if self.generators[1].active:
            print(self.generators[1].coefficients)

            self.generators[1].find_border()
            self.generators[1].calculate()

        if self.generators[2].active:
            print(self.generators[2].coefficients)

            self.generators[2].find_border()
            self.generators[2].calculate()

        self.rotate()


    def rotate(self) -> None:
        if self.generators[0].active:
            self.generators[0].coefficients = self.rotation @ self.generators[0].coefficients

        if self.generators[1].active:
            self.generators[1].coefficients = self.rotation @ self.generators[1].coefficients

        if self.generators[2].active:
            self.generators[2].coefficients = self.rotation @ self.generators[2].coefficients


    def get_pixel_array(self) -> np.ndarray:
        pixel_array = np.zeros((3, constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)

        if self.generators[0].active:
            pixel_histogram = (self.generators[0].histogram * 255).astype(np.uint8)
            pixel_array += self.generators[0].weight * pixel_histogram
        
        if self.generators[1].active:
            pixel_histogram = (self.generators[1].histogram * 255).astype(np.uint8)
            pixel_array += self.generators[1].weight * pixel_histogram

        if self.generators[2].active:
            pixel_histogram = (self.generators[2].histogram * 255).astype(np.uint8)
            pixel_array += self.generators[2].weight * pixel_histogram

        pixel_array = pixel_array.astype(np.uint8)
        pixel_array = pixel_array.transpose(1, 2, 0)

        return pixel_array
