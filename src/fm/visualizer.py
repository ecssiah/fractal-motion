from datetime import datetime
import os

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


    def render_border(self, border, label):
        frame_to_region_ratio = constants.FRAME_SIZE / constants.REGION_COUNT

        pixel_array = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)

        half_frame_size = constants.FRAME_SIZE // 2

        for y in range(half_frame_size):
            for x in range(constants.FRAME_SIZE):
                region_x = x // frame_to_region_ratio * constants.REGION_SIZE - constants.DOMAIN_RADIUS
                region_y = y // frame_to_region_ratio * constants.REGION_SIZE - constants.DOMAIN_RADIUS

                if (region_x, region_y) in border:
                    pixel_array[x, y] = 255
                    pixel_array[x, 2 * half_frame_size - y - 1] = 255

        imageio.imwrite(f'{self.directory}/borders/{label}.png', pixel_array)


    def render_frame(self, pixel_array, label):
        imageio.imwrite(f'{self.directory}/frames/{label}.png', pixel_array)


    def render_animation(self, pixel_arrays):
        filename = f'{self.directory}/fractal_{self.timestamp}.gif'

        imageio.mimsave(filename, pixel_arrays, duration=100, loop=0)
