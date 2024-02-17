#!/usr/bin/env python

import time

from fm import constants
from fm.transformer import Transformer
from fm.utils import time_string
from fm.visualizer import Visualizer

# TODO:
# 1. Add feature for rendering the borders into the animations


transformer = Transformer()
visualizer = Visualizer()


def main():
    start_time = time.time()

    run()

    elapsed_seconds = time.time() - start_time

    print(f'Run Time: { time_string(elapsed_seconds) }')


def run() -> None:
    pixel_arrays = []   

    for frame_index in range(1, constants.FRAME_COUNT + 1):
        visualizer.print_frame(frame_index)

        pixel_array = transformer.step()

        pixel_arrays.append(pixel_array)

        if frame_index % constants.DEBUG_INTERVAL == 0:
            visualizer.render_debug(transformer, frame_index, pixel_array)

    visualizer.render_animation(pixel_arrays)


if __name__ == '__main__':
    main()
