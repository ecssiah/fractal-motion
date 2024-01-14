import random
from typing import List, Tuple

import numpy as np

from fm import constants
from fm import chromogeometry
from fm.utils import print_percentage


class Generator:
    def __init__(self) -> None:
        self.active = True
        
        self.coefficients = np.array([1.0, 1.0, 1.0])
        self.exponents = np.array([4, 3, 2]).astype(int)

        self.counts = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)
        self.histogram = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)

        self.border_cells = []

        self.cell_corners = np.array([
            [ 0,                   0 ],
            [ constants.CELL_SIZE, 0 ],
            [ 0,                   constants.CELL_SIZE],
            [ constants.CELL_SIZE, constants.CELL_SIZE],
        ])


    def find_border(self) -> None:
        self.border_cells.clear()

        cell_corners = np.zeros(
            (constants.BORDER_MAP_SIZE + 1, (constants.BORDER_MAP_SIZE + 1) // 2 + 1), 
            dtype=np.uint8
        )

        for i in range(cell_corners.shape[0]):
            print_percentage(i, constants.BORDER_MAP_SIZE + 1, 'Corners')

            for j in range(cell_corners.shape[1]):
                C = chromogeometry.matrix(
                    constants.CELL_SIZE * i - constants.DOMAIN_RADIUS,
                    constants.CELL_SIZE * j - constants.DOMAIN_RADIUS,
                )

                if self.in_set(C):
                    cell_corners[i, j] = 1
        
        print_percentage(100, 100, 'Corners')
        print()

        cell_index = 0

        for center_x in np.arange(-constants.DOMAIN_RADIUS, constants.DOMAIN_RADIUS, constants.CELL_SIZE):
            cell_index += 1
            print_percentage(cell_index, constants.BORDER_MAP_SIZE + 1, 'Cells')

            for center_y in np.arange(-constants.DOMAIN_RADIUS, 0, constants.CELL_SIZE):
                number_of_escapes = 0

                for offset_x, offset_y in self.cell_corners:
                    x = center_x + offset_x
                    y = center_y + offset_y

                    i = int((x + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)
                    j = int((y + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)

                    number_of_escapes += cell_corners[i, j]

                if number_of_escapes > 0 and number_of_escapes < 4:
                    self.border_cells.append((x, y))
        
        print_percentage(100, 100, 'Cells')
        print()


    def calculate(self) -> None:
        self.counts.fill(0)

        for index in range(constants.POINTS):
            path = []

            z = C = self.get_border_seed()

            for _ in range(constants.ITERATIONS):
                z = self.apply_generator(z, C)

                if chromogeometry.quadrance(z) <= constants.ESCAPE_QUADRANCE:
                    path.append(z)
                else:
                    self.add_path_counts(path)
                    break

            if index % 1000 == 999 or index == constants.POINTS - 1:
                print_percentage(index, constants.POINTS, 'Paths')

        self.normalize()

        print()
        print()


    def apply_generator(self, z: np.ndarray, C: np.ndarray) -> np.ndarray:
        z = chromogeometry.conjugate(z)

        terms = [
            coefficient * np.linalg.matrix_power(z, exponent)
            for coefficient, exponent in zip(self.coefficients, self.exponents)
        ]

        return sum(terms) + C


    def add_path_counts(self, path: List[np.ndarray]) -> None:
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


    def normalize(self) -> None:
        max_value = np.max(self.counts)

        if (max_value > 0):
            self.histogram = np.log1p(self.counts) / np.log1p(max_value)
        else:
            self.histogram.fill(0.0)


    def set_coefficients(self, x: float, y: float, z: float) -> None:
        self.coefficients[:] = [x, y, z]


    def set_exponents(self, x: int, y: int, z: int) -> None:
        self.exponents[:] = [x, y, z]


    def get_random_seed(self) -> np.ndarray:
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, constants.DOMAIN_RADIUS + np.finfo(float).eps)

        a = radius * np.cos(angle)
        b = radius * np.sin(angle)

        return a * chromogeometry.IDENTITY + b * chromogeometry.BLUE


    def get_border_seed(self) -> np.ndarray:
        x, y = random.choice(self.border_cells)

        a = np.random.uniform(x, x + constants.CELL_SIZE)
        b = np.random.uniform(y, y + constants.CELL_SIZE)

        return a * chromogeometry.IDENTITY + b * chromogeometry.BLUE


    def in_set(self, C: np.ndarray) -> bool:
        z = C

        for _ in range(constants.ITERATIONS):
            z = self.apply_generator(z, C)

            if chromogeometry.quadrance(z) > constants.ESCAPE_QUADRANCE:
                return False
            
        return True


    def is_border(self, a: float, b: float) -> bool:
        num_of_escapes = sum(
            0 if self.in_set(a + x, b + y) else 1 
            for x, y in self.cell_corners
        )

        return num_of_escapes == 2 or num_of_escapes == 3


    def print_terms(self) -> None:
        print(
            f'f(z) = '
            f'{self.coefficients[0]:.2f}z{self.to_superscript(self.exponents[0])}'
            f' {"+" if self.coefficients[1] >= 0 else "-"} '
            f'{abs(self.coefficients[1]):.2f}z{self.to_superscript(self.exponents[1])}'
            f' {"+" if self.coefficients[2] >= 0 else "-"} '
            f'{abs(self.coefficients[2]):.2f}z{self.to_superscript(self.exponents[2])}'
            f' + C'
            f'\n'
        )


    def to_superscript(self, number):
        superscript_map = str.maketrans(
            '0123456789', 
            '⁰¹²³⁴⁵⁶⁷⁸⁹',
        )

        return str(number).translate(superscript_map)
    