#!/usr/bin/env python

import time

import numpy as np

from fm import constants
from fm.transformer import Transformer
from fm.utils import print_percentage, time_string
from fm.visualizer import Visualizer

# TODO:
# 1. Add feature for rendering the borders into the animations


transformer = Transformer()
visualizer = Visualizer()


def main():
    start_time = time.time()

    render_animation()

    elapsed_seconds = time.time() - start_time

    print(f'Execution Time: {time_string(elapsed_seconds)}')


def render_animation() -> None:
    pixel_arrays = []   

    for frame_index in range(constants.FRAME_COUNT):
        print_frame(frame_index)

        pixel_array = transformer.step()

        pixel_arrays.append(pixel_array)

        if frame_index % constants.DEBUG_INTERVAL == 0:
            render_debug(frame_index, pixel_array)

    visualizer.render_animation(pixel_arrays)


def render_debug(frame_index: int, pixel_array: np.ndarray) -> None:
    output = 0
    total_outputs = int(constants.DEBUG_FRAME) + transformer.mode.value

    frame_count_digits = len(str(constants.FRAME_COUNT))
    frame_index_output = f'{frame_index:0{frame_count_digits}d}'

    print_percentage(output, total_outputs, 'Debug')

    if constants.DEBUG_FRAME:
        output += 1
        
        visualizer.render_frame(pixel_array, label=f'frame_{frame_index_output}')
        
        print_percentage(output, total_outputs, 'Debug')

    if constants.DEBUG_BORDER:
        for index, generator in enumerate(transformer.generators):
            if generator.active:
                output += 1

                visualizer.render_border(generator.border_cells, label=f'border{index}_{frame_index_output}')

                print_percentage(output, total_outputs, 'Debug')
    
    print_percentage(100, 100, 'Debug')
    print()
    print()


def print_frame(index: int) -> None:
    padding = 4

    frame_count_digits = len(str(constants.FRAME_COUNT))
    frame_text = f'{" " * padding}FRAME {(index):0{frame_count_digits}d}/{constants.FRAME_COUNT - 1}{" " * padding}'

    print(f'§{"=" * len(frame_text)}§')
    print(f'§{frame_text}§')
    print(f'§{"=" * len(frame_text)}§')
    print()
    

if __name__ == '__main__':
    main()
