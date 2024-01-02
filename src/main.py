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
POINTS = 2 ** 16
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
        if point_index % 10000 == 0:
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
    os.makedirs(directory, exist_ok=True)

    frame_count = 10

    parameters = [
        Parameter(-1.0, 6),
        Parameter(1.0, 3)
    ]

    angle = 0.0
    pixel_arrays = []

    for frame_number in range(frame_count):
        parameters[0].coefficient = math.cos(angle)
        parameters[1].coefficient = math.sin(angle)

        angle += (2.0 * np.pi / frame_count)

        print(f'Frame: {frame_number}')

        frame_array = generate_frame(parameters)

        max_value = np.log1p(np.max(frame_array))

        frame_array_normalized = frame_array

        if (max_value > 0):
            frame_array_normalized = np.log1p(frame_array) / max_value

        brightness_array = (frame_array_normalized * 255).astype(np.uint8)

        pixel_array = np.stack([brightness_array] * 3, axis=2)
        pixel_arrays.append(pixel_array)

        if frame_number == 0:
            frame_filename = f'{directory}/first_frame.png'

            imageio.imwrite(frame_filename, pixel_array)

        print('100%\n')

    filename = f'{directory}/fractal_motion.gif'

    imageio.mimsave(filename, pixel_arrays, duration=100, loop=0)


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def elapsed_time_to_time_string(elapsed_seconds):
    minutes, seconds = divmod(elapsed_seconds, 60)

    return f"{int(minutes)}:{int(seconds):02d}"


def main():
    start_time = time.time()

    generate_animation()

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f'Execution Time: {elapsed_time_to_time_string(elapsed_time)}')
    

if __name__ == '__main__':
    main()
