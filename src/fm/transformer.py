import numpy as np
from scipy.spatial.transform import Rotation

from fm import constants
from fm.generator import Generator


class Transformer:
    def __init__(self) -> None:
        self.generators = [Generator() for _ in range(3)]

        self.generators[0].active = True
        self.generators[1].active = True
        self.generators[2].active = True

        self.generators[0].coefficients[:] = [0.0, 0.0, 1.0]
        self.generators[1].coefficients[:] = [0.0, 0.0, 1.0]
        self.generators[2].coefficients[:] = [0.0, 0.0, 1.0]

        self.generators[0].set_weight(1.0, 0.0, 0.0)
        self.generators[1].set_weight(0.0, 1.0, 0.0)
        self.generators[2].set_weight(0.0, 0.0, 1.0)

        self.angle = 2.0 * np.pi / constants.FRAME_COUNT

        self.axes = np.array([
            [ -1.0,  0.0,  0.0 ],
            [ -1.0,  1.0,  0.0 ],
            [  1.0,  1.0,  0.0 ],
        ])

        self.axes /= np.linalg.norm(self.axes, axis=1, keepdims=True)

        self.rotations = [Rotation.from_rotvec(self.angle * axis).as_matrix() for axis in self.axes]


    def step(self) -> None:
        for generator in self.generators:
            if generator.active:
                generator.print_terms()

                generator.find_border()
                generator.calculate()

        self.rotate()


    def rotate(self) -> None:
        for index, generator in enumerate(self.generators):
            if generator.active:
                generator.coefficients = self.rotations[index] @ generator.coefficients


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
