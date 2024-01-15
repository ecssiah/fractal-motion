import random
from typing import List, Tuple

import numpy as np

from fm import constants
from fm import chromogeometry
from fm.utils import print_percentage, to_superscript


class Generator:
    def __init__(self) -> None:
        self.active = True
        
        self.coefficients = np.array([1.0, 1.0, 1.0])
        self.exponents = np.array([4, 3, 2]).astype(int)

        self.counts = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)
        self.histogram = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)

        self.border_cells = []

        self.corner_offsets = np.array([
            [ -constants.CELL_RADIUS, -constants.CELL_RADIUS ],
            [ -constants.CELL_RADIUS,  constants.CELL_RADIUS ],
            [  constants.CELL_RADIUS, -constants.CELL_RADIUS ],
            [  constants.CELL_RADIUS,  constants.CELL_RADIUS ],
        ])


    def test_corners(self) -> np.ndarray:
        cell_corners = np.zeros(
            (constants.BORDER_MAP_SIZE + 1, (constants.BORDER_MAP_SIZE + 1) // 2 + 1), 
            dtype=np.uint8
        )

        for i in range(cell_corners.shape[0]):
            print_percentage(i, constants.BORDER_MAP_SIZE + 1, 'Corners')

            for j in range(cell_corners.shape[1]):
                x = i * constants.CELL_SIZE - constants.DOMAIN_RADIUS
                y = j * constants.CELL_SIZE - constants.DOMAIN_RADIUS

                C = chromogeometry.matrix_blue(x, y)

                if self.in_set(C):
                    cell_corners[i, j] = 1
        
        print_percentage(100, 100, 'Corners')
        print()

        return cell_corners
    

    def locate_border(self, cell_corners: np.ndarray) -> None:
        x_range = np.arange(
            -constants.DOMAIN_RADIUS + constants.CELL_RADIUS, 
             constants.DOMAIN_RADIUS - constants.CELL_RADIUS, 
            constants.CELL_SIZE
        )

        y_range = np.arange(
            -constants.DOMAIN_RADIUS + constants.CELL_RADIUS, 
             0                       - constants.CELL_RADIUS,
            constants.CELL_SIZE
        )

        for cell_index, center_x in enumerate(x_range):
            print_percentage(cell_index, constants.BORDER_MAP_SIZE + 1, 'Border')

            for center_y in y_range:
                number_of_escapes = 0

                for offset_x, offset_y in self.corner_offsets:
                    corner_x = center_x + offset_x
                    corner_y = center_y + offset_y

                    i = int((corner_x + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)
                    j = int((corner_y + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)

                    number_of_escapes += cell_corners[i, j]

                if number_of_escapes > 0 and number_of_escapes < 4:
                    self.border_cells.append((center_x, center_y))
        
        print_percentage(100, 100, 'Border')
        print()
    

    def calculate(self) -> None:
        self.counts.fill(0)
        self.border_cells.clear()

        cell_corners = self.test_corners()

        self.locate_border(cell_corners)

        for index in range(constants.POINTS):
            if index % 1000 == 0:
                print_percentage(index, constants.POINTS, 'Paths')

            path = []

            z = C = self.get_border_seed()

            for _ in range(constants.ITERATIONS):
                z = self.apply_generator(z, C)

                if chromogeometry.quadrance(z) <= constants.ESCAPE_QUADRANCE:
                    path.append(z)
                else:
                    self.add_path_counts(path)
                    break

        self.normalize()

        print_percentage(100, 100, 'Paths')
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
        for z in path:
            x = chromogeometry.real(z)
            y = chromogeometry.imag(z)

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

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        return chromogeometry.matrix_blue(x, y)


    def get_border_seed(self) -> np.ndarray:
        cell_x, cell_y = random.choice(self.border_cells)

        x = np.random.uniform(cell_x - constants.CELL_RADIUS, cell_x + constants.CELL_RADIUS)
        y = np.random.uniform(cell_y - constants.CELL_RADIUS, cell_y + constants.CELL_RADIUS)

        return chromogeometry.matrix_blue(x, y)


    def in_set(self, C: np.ndarray) -> bool:
        z = C

        for _ in range(constants.ITERATIONS):
            z = self.apply_generator(z, C)

            if chromogeometry.quadrance(z) > constants.ESCAPE_QUADRANCE:
                return False
            
        return True


    def print_terms(self) -> None:
        print(
            f'f(z) = '
            f'{self.coefficients[0]:.2f}z{to_superscript(self.exponents[0])}'
            f' {"+" if self.coefficients[1] >= 0 else "-"} '
            f'{abs(self.coefficients[1]):.2f}z{to_superscript(self.exponents[1])}'
            f' {"+" if self.coefficients[2] >= 0 else "-"} '
            f'{abs(self.coefficients[2]):.2f}z{to_superscript(self.exponents[2])}'
            f' + C'
        )

        print()
