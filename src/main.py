#!/usr/bin/env python

import time

import numpy as np

from fm import constants
from fm.transformer import Transformer
from fm.visualizer import Visualizer

# TODO:
# 1. Add feature for rendering the borders into the animations
# 2. Replace 2d rotations with complex multiplication to include dilations


pixel_arrays = []

transformer = Transformer()
visualizer = Visualizer()


def main():
    start_time = time.time()

    render_animation()

    elapsed_seconds = time.time() - start_time

    print(f'Execution Time: {get_time_string(elapsed_seconds)}')


def render_animation() -> None:
    for index in range(constants.FRAME_COUNT):
        print_frame(index)

        pixel_array = transformer.step()

        pixel_arrays.append(pixel_array)

        render_debug(index, pixel_array)

    visualizer.render_animation(pixel_arrays)


def render_debug(index: int, pixel_array: np.ndarray) -> None:
    if index % constants.DEBUG_INTERVAL == 0:
        if constants.DEBUG_FRAME:
            visualizer.render_frame(pixel_array, label=f'frame_{(index):06d}')

        if constants.DEBUG_BORDER:
            for index, generator in enumerate(transformer.generators):
                visualizer.render_border(generator.border, label=f'border_g{index}_{index:06d}')


def print_frame(index: int) -> None:
    print(f'<======================================>')
    print()
    print(f'FRAME {index + 1}/{constants.FRAME_COUNT}')
    print()


def get_time_string(elapsed_seconds: float) -> str:
    hours, seconds = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
    

if __name__ == '__main__':
    main()
