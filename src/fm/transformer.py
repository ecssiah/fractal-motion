import numpy as np
from scipy.spatial.transform import Rotation

from fm import constants
from fm.generator import Generator


class Transformer:
    def __init__(self) -> None:
        self.generators = [ Generator() for _ in range(3) ]

        self.generators[0].active = True
        self.generators[1].active = True
        self.generators[2].active = True

        self.generators[0].coefficients[:] = [1.0, 0.0, 0.0]
        self.generators[1].coefficients[:] = [0.0, 1.0, 0.0]
        self.generators[2].coefficients[:] = [0.0, 0.0, 1.0]

        self.generators[0].exponents[:] = [5, 4, 3]
        self.generators[1].exponents[:] = [5, 4, 3]
        self.generators[2].exponents[:] = [5, 4, 3]

        self.weights = np.array([
            [ 0.0, 0.0, 1.0 ],
            [ 0.0, 1.0, 0.0 ],
            [ 1.0, 0.0, 0.0 ],
        ])

        self.angle = 2.0 * np.pi / constants.FRAME_COUNT

        self.axes = np.array([
            [  1.0,  1.0,  1.0 ],
            [  1.0,  1.0,  1.0 ],
            [  1.0,  1.0,  1.0 ],
        ])

        self.axes /= np.linalg.norm(self.axes, axis=1, keepdims=True)

        self.rotations = [ 
            Rotation.from_rotvec(self.angle * axis).as_matrix() 
            for axis in self.axes 
        ]


    def step(self) -> np.ndarray:
        for index, generator in enumerate(self.generators):
            if generator.active:
                generator.print_terms()

                generator.find_border()
                generator.calculate()

                generator.coefficients = self.rotations[index] @ generator.coefficients

        return self.get_pixel_array()


    def get_pixel_array(self) -> np.ndarray:
        histogram_combined = np.zeros(
            (constants.FRAME_SIZE, constants.FRAME_SIZE, 3), 
            dtype=np.float64
        )

        for generator, weight in zip(self.generators, self.weights):
            if generator.active:
                histogram_combined += generator.histogram[:, :, np.newaxis] * weight
        
        return (histogram_combined * 255).astype(np.uint8)
