#!/usr/bin/env python

import math
import time

import numpy as np

from fm import constants
from fm.generator import Generator
from fm.parameter import Parameter
from fm.visualizer import Visualizer


def generate_animation():
    visualizer = Visualizer()

    parameters = [
        Parameter(-1.0, 4)
    ]

    angle = 0.0
    pixel_arrays = []

    for frame_number in range(constants.FRAME_COUNT):
        parameters[0].coefficient = math.cos(angle)

        angle += (2.0 * np.pi / constants.FRAME_COUNT)

        print(f'Frame: {frame_number}')

        generator = Generator(parameters)
        generator.run()

        histogram_pixel = (generator.histogram * 255).astype(np.uint8)

        pixel_array = np.stack([histogram_pixel] * 3, axis=2)
        pixel_arrays.append(pixel_array)

        if frame_number == 0:
            visualizer.generate_frame(pixel_array)

        print('100%\n')

    visualizer.generate_animation(pixel_arrays)


def elapsed_time_to_time_string(elapsed_seconds):
    minutes, seconds = divmod(elapsed_seconds, 60)

    return f"{int(minutes)}:{int(seconds):02d}"


def main():
    start_time = time.time()

    generate_animation()

    elapsed_time = time.time() - start_time

    print(f'Execution Time: {elapsed_time_to_time_string(elapsed_time)}')
    

if __name__ == '__main__':
    main()
