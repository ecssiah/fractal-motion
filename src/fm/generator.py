import random
from typing import List, Tuple

import numpy as np

from fm import constants
from fm import chromogeometry
from fm.parameter import Parameter


class Generator:
    def __init__(self, parameters: List[Parameter]) -> None:
        self.border = []
        self.parameters = parameters
        self.weights = np.array([1.0, 0.0, 0.0])[:, np.newaxis, np.newaxis]

        self.counts = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)
        self.histogram = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)


    def set_weights(self, red: float, green: float, blue: float) -> None:
        sum = red + green + blue

        self.weights[0, 0, 0] = red / sum
        self.weights[1, 0, 0] = green / sum
        self.weights[2, 0, 0] = blue / sum


    def in_set(self, a: float, b: float) -> bool:
        C = a * chromogeometry.IDENTITY + b * chromogeometry.BLUE

        z = C

        for _ in range(constants.ITERATIONS):
            z_adjugate = chromogeometry.adjugate(z)
            z = 0

            for parameter in self.parameters:
                z += parameter.coefficient * np.linalg.matrix_power(z_adjugate, parameter.exponent)

            z += C

            quadrance = chromogeometry.quadrance(z)

            if quadrance > constants.ESCAPE_QUADRANCE:
                return False
            
        return True


    def find_border(self) -> None:
        print('Calculating Borders')

        half_region_count = constants.REGION_COUNT // 2 + 1

        for j in range(half_region_count):
            percent_complete = j / half_region_count * 100
            print(f'\r{percent_complete:.1f}% ', end='', flush=True)

            for i in range(constants.REGION_COUNT):
                a = constants.REGION_SIZE * i - constants.DOMAIN_RADIUS
                b = constants.REGION_SIZE * j - constants.DOMAIN_RADIUS

                corners = [
                    (0,                     0),
                    (constants.REGION_SIZE, 0),
                    (0,                     constants.REGION_SIZE),
                    (constants.REGION_SIZE, constants.REGION_SIZE)
                ]

                num_of_escapes = sum(0 if self.in_set(a + x, b + y) else 1 for x, y in corners)

                if num_of_escapes == 2 or num_of_escapes == 3:
                    self.border.append((a,  b))
                    self.border.append((a, -b))

        print('\r100.0% ', end='', flush=True)
        print()


    def get_random_seed(self) -> Tuple[float, float]:
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, constants.DOMAIN_RADIUS + np.finfo(float).eps)
        
        a = radius * np.cos(angle)
        b = radius * np.sin(angle)

        return a, b


    def get_border_seed(self) -> Tuple[float, float]:
        region = random.choice(self.border)

        a = np.random.uniform(region[0], region[0] + constants.REGION_SIZE)
        b = np.random.uniform(region[1], region[1] + constants.REGION_SIZE)

        return a, b


    def calculate(self) -> None:
        print('Calculating Paths')

        for point_index in range(constants.POINTS):
            if point_index % 2000 == 0:
                percent_complete = point_index / constants.POINTS * 100
                print(f'\r{percent_complete:.1f}% ', end='', flush=True)

            path = []

            a, b = self.get_border_seed()

            C = a * chromogeometry.IDENTITY + b * chromogeometry.BLUE

            z = C

            for _ in range(constants.ITERATIONS):
                z_adjugate = chromogeometry.adjugate(z)
                z = 0

                for parameter in self.parameters:
                    z += parameter.coefficient * np.linalg.matrix_power(z_adjugate, parameter.exponent)

                z += C

                if chromogeometry.quadrance(z) <= constants.ESCAPE_QUADRANCE:
                    path.append(z)
                else:
                    for point in path:
                        x, y = point[0]

                        cell_x = int(
                            (x + constants.DOMAIN_RADIUS) / constants.DOMAIN_SIZE * (constants.FRAME_SIZE - 1)
                        )
                        cell_y = int(
                            (y + constants.DOMAIN_RADIUS) / constants.DOMAIN_SIZE * (constants.FRAME_SIZE - 1)
                        )

                        in_x_bounds = cell_x >= 0 and cell_x < constants.FRAME_SIZE
                        in_y_bounds = cell_y >= 0 and cell_y < constants.FRAME_SIZE

                        if in_x_bounds and in_y_bounds:
                            symmetric_cell_y = int(
                                (-y + constants.DOMAIN_RADIUS) / constants.DOMAIN_SIZE * (constants.FRAME_SIZE - 1)
                            )
                            
                            self.counts[cell_x, cell_y] += 1
                            self.counts[cell_x, symmetric_cell_y] += 1

                    break

        self.normalize()

        print('\r100.0% ', end='', flush=True)
        print()


    def normalize(self) -> None:
        self.max_value = np.max(self.counts)

        if (self.max_value > 0):
            self.histogram = np.log1p(self.counts) / np.log1p(self.max_value)
        else:
            self.histogram.fill(0.0)
