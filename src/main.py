#!/usr/bin/env python

import math
import time

import numpy as np

from fm import constants
from fm.generator import Generator
from fm.parameter import Parameter
from fm.visualizer import Visualizer


def render_animation():
    visualizer = Visualizer()

    parameters1 = [
        Parameter( 1.0, 4),
        Parameter( 1.0, 2)
    ]

    parameters2 = [
        Parameter( 1.0, 5),
        Parameter( 1.0, 3)
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
        angle2 += (2.0 * np.pi / constants.FRAME_COUNT)

        print(f'\nFrame {frame_number + 1}/{constants.FRAME_COUNT}')

        generator1 = Generator(parameters1)
        generator1.set_weights(0.75, 0.25, 0.0)
        generator1.find_border()
        generator1.calculate()

        generator2 = Generator(parameters2)
        generator2.set_weights(0.0, 0.25, 0.75)
        generator2.find_border()
        generator2.calculate()
        
        histogram_pixel1 = (generator1.histogram * 255).astype(np.uint8)
        histogram_pixel2 = (generator2.histogram * 255).astype(np.uint8)

        pixel_array = (
            generator1.weights * histogram_pixel1 + 
            generator2.weights * histogram_pixel2
        )

        pixel_array = pixel_array.astype(np.uint8)
        pixel_array = pixel_array.transpose(1, 2, 0)

        pixel_arrays.append(pixel_array)

        if frame_number % 10 == 0:
            # visualizer.render_border(generator1.border, label=f'gen1_{frame_number:06d}')
            # visualizer.render_border(generator2.border, label=f'gen2_{frame_number:06d}')

            visualizer.render_frame(pixel_array, label=f'{(frame_number + 1):06d}')

    visualizer.render_animation(pixel_arrays)


def elapsed_time_to_time_string(elapsed_seconds):
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"


def main():
    start_time = time.time()

    render_animation()

    elapsed_time = time.time() - start_time

    print(f'Execution Time: {elapsed_time_to_time_string(elapsed_time)}')
    

if __name__ == '__main__':
    main()
