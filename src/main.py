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

    parameters1 = [
        Parameter( 1.0, 3),
        Parameter( 1.0, 2)
    ]

    parameters2 = [
        Parameter( 1.0, 3),
        Parameter( 1.0, 2)
    ]

    angle1 = 0.0
    angle2 = 0.0

    pixel_arrays = []

    for frame_number in range(constants.FRAME_COUNT):
        parameters1[0].coefficient = math.cos(angle1)
        parameters1[1].coefficient = math.sin(angle1)

        parameters2[0].coefficient = math.cos(angle2)
        parameters2[1].coefficient = math.sin(angle2)

        angle1 += (2.0 * np.pi / constants.FRAME_COUNT)
        angle2 -= (2.0 * np.pi / constants.FRAME_COUNT)

        print(f'\nFrame {frame_number + 1}/{constants.FRAME_COUNT}')

        generator1 = Generator(parameters1)
        generator1.run()

        generator2 = Generator(parameters2)
        generator2.run()

        histogram_pixel1 = (generator1.histogram * 255).astype(np.uint8)
        histogram_pixel2 = (generator2.histogram * 255).astype(np.uint8)

        weights_generator1 = np.array([0.5, 0.0, 0.5])[:, np.newaxis, np.newaxis]
        weights_generator2 = np.array([0.0, 0.5, 0.5])[:, np.newaxis, np.newaxis]

        pixel_array = (
            weights_generator1 * histogram_pixel1 +
            weights_generator2 * histogram_pixel2
        ).astype(np.uint8)

        pixel_array = pixel_array.transpose(1, 2, 0)

        pixel_arrays.append(pixel_array)

        if frame_number == 0:
            visualizer.generate_frame(pixel_array)

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
