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


    def render_border(self, borders: List[Tuple[float, float]], label: str) -> None:
        half_frame_size = constants.FRAME_SIZE // 2
        frame_to_region_ratio = constants.FRAME_SIZE / constants.REGION_COUNT

        pixel_array = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)

        for y in range(half_frame_size):
            for x in range(constants.FRAME_SIZE):
                region_x = x // frame_to_region_ratio * constants.REGION_SIZE - constants.DOMAIN_RADIUS
                region_y = y // frame_to_region_ratio * constants.REGION_SIZE - constants.DOMAIN_RADIUS

                symmetric_y = 2 * half_frame_size - (y + 1)

                if (region_x, region_y) in borders:
                    pixel_array[x, y] = 255
                    pixel_array[x, symmetric_y] = 255

        imageio.imwrite(f'{self.directory}/borders/{label}.png', pixel_array)


    def render_frame(self, pixel_array: np.ndarray, label: str):
        imageio.imwrite(f'{self.directory}/frames/{label}.png', pixel_array)


    def render_animation(self, pixel_arrays: List[np.ndarray]):
        filename = f'{self.directory}/fractal_{self.timestamp}.gif'

        imageio.mimsave(filename, pixel_arrays, duration=100, loop=0)
