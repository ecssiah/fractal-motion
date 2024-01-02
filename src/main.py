#!/usr/bin/env python

import os
import math
import time
from datetime import datetime
from typing import List
import numpy as np
import imageio

from fm.parameter import Parameter

ESCAPE_RADIUS_SQUARED = 2.0 ** 2
POINTS = 2 ** 20
ITERATIONS = 2 ** 5
FRAME_SIZE = 800
DOMAIN_SIZE = 4.0
DOMAIN_RADIUS = DOMAIN_SIZE / 2.0

I = np.array([
    [ 1,  0],
    [ 0,  1],
])

R = np.array([
    [ 0,  1],
    [ 1,  0],
])

G = np.array([
    [ 1,  0],
    [ 0, -1],
])

B = np.array([
    [ 0,  1],
    [-1,  0],
])


def conjugate(z):
    z[0, 1] *= -1
    z[1, 0] *= -1

    return z


def get_seed():
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, DOMAIN_RADIUS + np.finfo(float).eps)
    
    a = radius * np.cos(angle)
    b = radius * np.sin(angle)

    return (a, b)


def generate_frame(parameters: List[Parameter]):
    frame_array = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

    for point_index in range(POINTS):
        if point_index % 50000 == 0:
            print(f"{int(point_index / POINTS * 100)}%")

        path = []

        a, b = get_seed()

        C = a * I + b * B

        z = C

        for _ in range(ITERATIONS):
            z_conjugate = conjugate(z)
            z = 0

            for parameter in parameters:
                z += parameter.coefficient * np.linalg.matrix_power(z_conjugate, parameter.exponent)

            z += C

            magnitude_squared = np.sum(np.square(z[:, 0]))

            if magnitude_squared <= ESCAPE_RADIUS_SQUARED:
                path.append(z)
            else:
                for point in path:
                    x = point[0, 0]
                    y = point[0, 1]

                    cell_x = int((x + DOMAIN_RADIUS) / DOMAIN_SIZE * (FRAME_SIZE - 1))
                    cell_y = int((y + DOMAIN_RADIUS) / DOMAIN_SIZE * (FRAME_SIZE - 1))

                    in_x_bounds = cell_x >= 0 and cell_x < FRAME_SIZE
                    in_y_bounds = cell_y >= 0 and cell_y < FRAME_SIZE

                    if in_x_bounds and in_y_bounds:
                        symmetric_cell_y = int((-y + DOMAIN_RADIUS) / DOMAIN_SIZE * (FRAME_SIZE - 1))
                        
                        frame_array[cell_x, cell_y] += 1
                        frame_array[cell_x, symmetric_cell_y] += 1

                break

    return frame_array


def generate_animation():
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

    directory = f'output/{timestamp}'
    os.makedirs(directory + '/frames', exist_ok=True)

    parameters = [
        Parameter(coefficient=1.0, exponent=4),
        Parameter(coefficient=0.0, exponent=2)
    ]

    pixel_arrays = []

    angle = 0.0
    frame_count = 200

    for frame in range(frame_count):
        parameters[0].coefficient = math.cos(angle)
        parameters[1].coefficient = math.sin(angle)

        angle += (2.0 * np.pi / frame_count)

        frame_array = generate_frame(parameters)

        frame_array_normalized = frame_array

        max_value = np.log1p(np.max(frame_array))

        if (max_value > 0):
            frame_array_normalized = np.log1p(frame_array) / max_value

        brightness_array = (frame_array_normalized * 255).astype(np.uint8)

        pixel_array = np.stack([brightness_array] * 3, axis=2)

        frame_filename = f'{directory}/frames/frame_{frame:04d}.png'
        imageio.imwrite(frame_filename, pixel_array)

        pixel_arrays.append(frame)

    filename = f'{directory}/fractal_motion.gif'

    frames = []

    for frame in range(len(pixel_arrays)):
        frames.append(imageio.v2.imread(f'{directory}/frames/frame_{frame:04d}.png'))

    imageio.mimsave(filename, frames, duration=100, loop=0)


def main():
    start_time = time.time()

    generate_animation()

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f'Execution Time: {elapsed_time:.2f} s')
    

if __name__ == '__main__':
    main()
