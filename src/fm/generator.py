import random
from typing import List, Tuple

import numpy as np

from fm import constants, labels
from fm import chromogeometry
from fm.chromogeometry import Color
from fm.utils import print_percentage, to_superscript


class Generator:
    def __init__(self) -> None:
        self.active = True
        
        self.coefficients = np.array([1.0, 1.0, 1.0])
        self.exponents = np.array([4, 3, 2]).astype(int)

        self.counts = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)
        self.histogram = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)
        self.cell_corners = np.zeros((constants.BORDER_MAP_SIZE + 1, (constants.BORDER_MAP_SIZE + 1) // 2 + 1), dtype=np.uint8)

        self.corner_offsets = np.array([
            [ -constants.CELL_RADIUS, -constants.CELL_RADIUS ],
            [ -constants.CELL_RADIUS,  constants.CELL_RADIUS ],
            [  constants.CELL_RADIUS, -constants.CELL_RADIUS ],
            [  constants.CELL_RADIUS,  constants.CELL_RADIUS ],
        ])

        self.border_cells = []


    def test_corners(self) -> np.ndarray:
        self.cell_corners.fill(0)

        for i in range(self.cell_corners.shape[0]):
            print_percentage(i, constants.BORDER_MAP_SIZE + 1, labels.CORNERS)

            for j in range(self.cell_corners.shape[1]):
                x = i * constants.CELL_SIZE - constants.DOMAIN_RADIUS
                y = j * constants.CELL_SIZE - constants.DOMAIN_RADIUS

                C = chromogeometry.matrix(x, y, Color.BLUE)

                if self.in_set(C):
                    self.cell_corners[i, j] = 1
        
        print_percentage(100, 100, labels.CORNERS)
        print()
    

    def locate_border(self) -> None:
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
            print_percentage(cell_index, constants.BORDER_MAP_SIZE + 1, labels.BORDERS)

            for center_y in y_range:
                number_of_escapes = 0

                for offset_x, offset_y in self.corner_offsets:
                    corner_x = center_x + offset_x
                    corner_y = center_y + offset_y

                    i = int((corner_x + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)
                    j = int((corner_y + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)

                    number_of_escapes += self.cell_corners[i, j]

                if number_of_escapes > 0 and number_of_escapes < 4:
                    self.border_cells.append((center_x, center_y))
        
        print_percentage(100, 100, labels.BORDERS)
        print()
    

    def calculate(self) -> None:
        self.counts.fill(0)
        self.border_cells.clear()

        self.test_corners()
        self.locate_border()

        for index in range(constants.POINTS):
            if index % 1000 == 0:
                print_percentage(index, constants.POINTS, labels.PATHS)

            path = []

            z = C = self.get_border_seed()

            for _ in range(constants.ITERATIONS):
                z = self.iterate_function(z, C)

                if chromogeometry.quadrance(z) <= constants.ESCAPE_QUADRANCE:
                    path.append(z)
                else:
                    self.add_path_counts(path)
                    break

        self.normalize()

        print_percentage(100, 100, labels.PATHS)
        print()
        print()


    def iterate_function(self, z: np.ndarray, C: np.ndarray) -> np.ndarray:
        z = chromogeometry.conjugate(z)

        terms = [
            coefficient * np.linalg.matrix_power(z, exponent)
            for coefficient, exponent in zip(self.coefficients, self.exponents)
        ]

        return sum(terms) + C


    def add_path_counts(self, path: List[np.ndarray]) -> None:
        for z in path:
            x = chromogeometry.x_component(z, Color.BLUE)
            y = chromogeometry.y_component(z, Color.BLUE)

            cell_x, cell_y = self.calculate_cell_coordinates(x, y)

            in_x_bounds = 0 <= cell_x < constants.FRAME_SIZE
            in_y_bounds = 0 <= cell_y < constants.FRAME_SIZE

            if in_x_bounds and in_y_bounds:
                _, cell_symmetric_y = self.calculate_cell_coordinates(x, -y)
                
                self.counts[cell_x, cell_y] += 1
                self.counts[cell_x, cell_symmetric_y] += 1

    
    def calculate_cell_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        centered_x = x + constants.DOMAIN_RADIUS
        centered_y = y + constants.DOMAIN_RADIUS

        normalized_x = centered_x / constants.DOMAIN_SIZE
        normalized_y = centered_y / constants.DOMAIN_SIZE

        cell_x = int(normalized_x * (constants.FRAME_SIZE - 1))
        cell_y = int(normalized_y * (constants.FRAME_SIZE - 1))

        return cell_x, cell_y


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

        return chromogeometry.matrix(x, y, Color.BLUE)


    def get_border_seed(self) -> np.ndarray:
        cell_x, cell_y = random.choice(self.border_cells)



        min_x = cell_x - constants.CELL_RADIUS
        max_x = cell_x + constants.CELL_RADIUS
        min_y = cell_y - constants.CELL_RADIUS
        max_y = cell_y + constants.CELL_RADIUS

        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        return chromogeometry.matrix(x, y, Color.BLUE)


    def in_set(self, C: np.ndarray) -> bool:
        z = C

        for _ in range(constants.ITERATIONS):
            z = self.iterate_function(z, C)

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
