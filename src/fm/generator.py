import random
from typing import List, Tuple

import numpy as np

from fm import constants
from fm import chromogeometry


class Generator:
    def __init__(self, weight: List[float] = [1.0, 1.0, 1.0]) -> None:
        self.active = True

        self.coefficients = np.array([1.0, 1.0, 1.0])
        self.exponents = np.array([4, 3, 2]).astype(int)

        self.border = []
        self.weight = np.array(weight)[:, np.newaxis, np.newaxis]

        self.counts = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)
        self.histogram = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)


    def set_weight(self, red: float, green: float, blue: float) -> None:
        sum = red + green + blue

        self.weight[0, 0, 0] = red / sum
        self.weight[1, 0, 0] = green / sum
        self.weight[2, 0, 0] = blue / sum


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


    def in_set(self, a: float, b: float) -> bool:
        z = C = a * chromogeometry.IDENTITY + b * chromogeometry.BLUE

        for _ in range(constants.ITERATIONS):
            z_conjugate = chromogeometry.conjugate(z)

            z = (
                self.coefficients[0] * np.linalg.matrix_power(z_conjugate, self.exponents[0]) +
                self.coefficients[1] * np.linalg.matrix_power(z_conjugate, self.exponents[1]) +
                self.coefficients[2] * np.linalg.matrix_power(z_conjugate, self.exponents[2]) +
                C
            )

            quadrance = chromogeometry.quadrance(z)

            if quadrance > constants.ESCAPE_QUADRANCE:
                return False
            
        return True


    def find_border(self) -> None:
        print('Borders')

        self.border.clear()

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


    def calculate(self) -> None:
        print('Paths')

        self.counts.fill(0)

        for point_index in range(constants.POINTS):
            if point_index % 2000 == 0:
                percent_complete = point_index / constants.POINTS * 100
                print(f'\r{percent_complete:.1f}% ', end='', flush=True)

            path = []

            a, b = self.get_border_seed()

            z = C = a * chromogeometry.IDENTITY + b * chromogeometry.BLUE

            for _ in range(constants.ITERATIONS):
                z_conjugate = chromogeometry.conjugate(z)

                z = (
                    self.coefficients[0] * np.linalg.matrix_power(z_conjugate, self.exponents[0]) +
                    self.coefficients[1] * np.linalg.matrix_power(z_conjugate, self.exponents[1]) +
                    self.coefficients[2] * np.linalg.matrix_power(z_conjugate, self.exponents[2]) +
                    C
                )

                if chromogeometry.quadrance(z) <= constants.ESCAPE_QUADRANCE:
                    path.append(z)
                else:
                    for point in path:
                        x, y = point[0]

                        centered_x = x + constants.DOMAIN_RADIUS
                        centered_y = y + constants.DOMAIN_RADIUS

                        normalized_x = centered_x / constants.DOMAIN_SIZE
                        normalized_y = centered_y / constants.DOMAIN_SIZE

                        cell_x = int(normalized_x * (constants.FRAME_SIZE - 1))
                        cell_y = int(normalized_y * (constants.FRAME_SIZE - 1))

                        in_x_bounds = cell_x >= 0 and cell_x < constants.FRAME_SIZE
                        in_y_bounds = cell_y >= 0 and cell_y < constants.FRAME_SIZE

                        if in_x_bounds and in_y_bounds:
                            centered_symmetric_y = -y + constants.DOMAIN_RADIUS

                            normalized_symmetric_y = centered_symmetric_y / constants.DOMAIN_SIZE

                            cell_symmetric_y = int(normalized_symmetric_y * (constants.FRAME_SIZE - 1))
                            
                            self.counts[cell_x, cell_y] += 1
                            self.counts[cell_x, cell_symmetric_y] += 1

                    break

        self.normalize()

        print('\r100.0% ', end='', flush=True)
        print()
        print()


    def normalize(self) -> None:
        self.max_value = np.max(self.counts)

        if (self.max_value > 0):
            self.histogram = np.log1p(self.counts) / np.log1p(self.max_value)
        else:
            self.histogram.fill(0.0)


    def print_terms(self) -> None:
        print(
            f'ƒ(z) = '
            f'{self.coefficients[0]:.2f}z{self.to_superscript(self.exponents[0])}'
            f' {"+" if self.coefficients[1] > 0 else "-"} '
            f'{abs(self.coefficients[1]):.2f}z{self.to_superscript(self.exponents[1])}'
            f' {"+" if self.coefficients[2] > 0 else "-"} '
            f'{abs(self.coefficients[2]):.2f}z{self.to_superscript(self.exponents[2])}'
            f' + C'
        )


    def to_superscript(self, number):
        superscript_map = str.maketrans(
            '0123456789', 
            '⁰¹²³⁴⁵⁶⁷⁸⁹',
        )

        return str(number).translate(superscript_map)
    