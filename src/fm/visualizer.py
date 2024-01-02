from datetime import datetime
import os

import imageio

class Visualizer:
    def __init__(self) -> None:
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.directory = f'output/{self.timestamp}'

        os.makedirs(self.directory, exist_ok=True)


    def generate_frame(self, pixel_array):
        filename = f'{self.directory}/test_frame.png'

        imageio.imwrite(filename, pixel_array)


    def generate_animation(self, pixel_arrays):
        filename = f'{self.directory}/fractal_motion_{self.timestamp}.gif'

        imageio.mimsave(filename, pixel_arrays, duration=100, loop=0)
