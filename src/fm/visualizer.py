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


    def generate_border_regions(self, border_regions, frame_number=None):
        frame_to_region_ratio = constants.FRAME_SIZE / constants.REGION_COUNT

        pixel_array = np.zeros((constants.FRAME_SIZE, constants.FRAME_SIZE), dtype=np.uint8)

        for y in range(constants.FRAME_SIZE):
            for x in range(constants.FRAME_SIZE):
                region_x = x // frame_to_region_ratio * constants.REGION_SIZE - constants.DOMAIN_RADIUS
                region_y = y // frame_to_region_ratio * constants.REGION_SIZE - constants.DOMAIN_RADIUS

                if (region_x, region_y) in border_regions:
                    pixel_array[x, y] = 255

        if frame_number == None:
            imageio.imwrite(f'{self.directory}/border_regions.png', pixel_array)
        else:
            imageio.imwrite(f'{self.directory}/border_regions_{frame_number}.png', pixel_array)


    def generate_frame(self, pixel_array, frame_number=None):
        if frame_number == None:
            imageio.imwrite(f'{self.directory}/frame.png', pixel_array)
        else:
            imageio.imwrite(f'{self.directory}/frame_{frame_number:04d}.png', pixel_array)


    def generate_animation(self, pixel_arrays):
        filename = f'{self.directory}/fractal_motion_{self.timestamp}.gif'

        imageio.mimsave(filename, pixel_arrays, duration=100, loop=0)
