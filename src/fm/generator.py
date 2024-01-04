import math
import random
from typing import List

import numpy as np

from fm import constants
from fm import chromogeometry
from fm.parameter import Parameter

class Generator:
    def __init__(self, parameters: List[Parameter]) -> None:
        self.border_regions = []
        self.parameters = parameters

        self.counts = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)
        self.histogram = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.float64)

        self.find_border_regions()


    def test_escape(self, a, b):
        C = a * chromogeometry.IDENTITY + b * chromogeometry.BLUE

        z = C

        for _ in range(constants.ITERATIONS):
            z_conjugate = chromogeometry.conjugate(z)
            z = 0

            for parameter in self.parameters:
                z += parameter.coefficient * np.linalg.matrix_power(z_conjugate, parameter.exponent)

            z += C

            magnitude_squared = np.sum(np.square(z[:, 0]))

            if magnitude_squared > constants.ESCAPE_RADIUS_SQUARED:
                return True
            
        return False


    def find_border_regions(self):
        print('Finding Border Regions')

        for j in range(constants.REGION_COUNT):
            percent_complete = j / constants.REGION_COUNT * 100
            print(f'\r{percent_complete:.1f}% ', end='', flush=True)

            for i in range(constants.REGION_COUNT):
                a = constants.REGION_SIZE * i - constants.DOMAIN_RADIUS
                b = constants.REGION_SIZE * j - constants.DOMAIN_RADIUS

                num_of_escapes = 0

                num_of_escapes += 1 if self.test_escape(a, b) else 0
                num_of_escapes += 1 if self.test_escape(a + constants.REGION_SIZE, b) else 0
                num_of_escapes += 1 if self.test_escape(a, b + constants.REGION_SIZE) else 0
                num_of_escapes += 1 if self.test_escape(a + constants.REGION_SIZE, b + constants.REGION_SIZE) else 0

                if num_of_escapes == 2 or num_of_escapes == 3:
                    self.border_regions.append((a, b))

        print('\r100.0% ', end='', flush=True)
        print()


    def get_seed(self):
        region = random.choice(self.border_regions)

        a = np.random.uniform(region[0], region[0] + constants.REGION_SIZE)
        b = np.random.uniform(region[1], region[1] + constants.REGION_SIZE)

        return a, b
    
    
    # def get_seed(self):
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     radius = np.random.uniform(0, constants.DOMAIN_RADIUS + np.finfo(float).eps)
        
    #     a = radius * np.cos(angle)
    #     b = radius * np.sin(angle)

    #     return a, b


    def run(self):
        self.calculate()
        self.normalize()

    
    def calculate(self):
        print('Calculating Paths')

        for point_index in range(constants.POINTS):
            if point_index % 10000 == 0:
                percent_complete = point_index / constants.POINTS * 100
                print(f'\r{percent_complete:.1f}% ', end='', flush=True)

            path = []

            a, b = self.get_seed()

            C = a * chromogeometry.IDENTITY + b * chromogeometry.BLUE

            z = C

            for _ in range(constants.ITERATIONS):
                z_conjugate = chromogeometry.conjugate(z)
                z = 0

                for parameter in self.parameters:
                    z += parameter.coefficient * np.linalg.matrix_power(z_conjugate, parameter.exponent)

                z += C

                magnitude_squared = np.sum(np.square(z[:, 0]))

                if magnitude_squared <= constants.ESCAPE_RADIUS_SQUARED:
                    path.append(z)
                else:
                    for point in path:
                        x = point[0, 0]
                        y = point[0, 1]

                        cell_x = int(
                            (x + constants.DOMAIN_RADIUS) / constants.DOMAIN_SIZE * (constants.FRAME_SIZE - 1)
                        )
                        cell_y = int(
                            (y + constants.DOMAIN_RADIUS) / constants.DOMAIN_SIZE * (constants.FRAME_SIZE - 1)
                        )

                        in_x_bounds = cell_x >= 0 and cell_x < constants.FRAME_SIZE
                        in_y_bounds = cell_y >= 0 and cell_y < constants.FRAME_SIZE

                        if in_x_bounds and in_y_bounds:
                            symmetric_cell_y = int((-y + constants.DOMAIN_RADIUS) / constants.DOMAIN_SIZE * (constants.FRAME_SIZE - 1))
                            
                            self.counts[cell_x, cell_y] += 1
                            self.counts[cell_x, symmetric_cell_y] += 1

                    break

        print('\r100.0% ', end='', flush=True)
        print()


    def normalize(self):
        self.max_value = np.max(self.counts)

        if (self.max_value > 0):
            self.histogram = np.log1p(self.counts) / np.log1p(self.max_value)
        else:
            self.histogram.fill(0.0)
