from datetime import datetime
import os
from typing import List, Tuple

import imageio
import numpy as np

from fm import constants


class Visualizer:
    def __init__(self) -> None:
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.directory = f'output/{self.timestamp}'

        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(f'{self.directory}/frames', exist_ok=True)
        os.makedirs(f'{self.directory}/borders', exist_ok=True)


    def render_border(self, border_cells: List[Tuple[float, float]], label: str) -> None:
        pixel_array = np.zeros((constants.BORDER_MAP_SIZE, constants.BORDER_MAP_SIZE), dtype=np.uint8)

        for cell_x, cell_y in border_cells:
            x = int((cell_x + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)
            y = int((cell_y + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)

            symmetric_y = 2 * constants.BORDER_MAP_RADIUS - 1 - y

            pixel_array[x, y] = 255
            pixel_array[x, symmetric_y] = 255

        imageio.imwrite(f'{self.directory}/borders/{label}.png', pixel_array)


    def render_frame(self, pixel_array: np.ndarray, label: str):
        imageio.imwrite(f'{self.directory}/frames/{label}.png', pixel_array)


    def render_animation(self, pixel_arrays: List[np.ndarray]):
        filename = f'{self.directory}/fractal_{self.timestamp}.gif'

        imageio.mimsave(filename, pixel_arrays, duration=100, loop=0)
