#!/usr/bin/env python

import time

from fm import constants
from fm.transformer import Transformer
from fm.visualizer import Visualizer

# TODO:
# 1. Add feature for rendering the borders into the animations
# 2. Replace 2d rotations with complex multiplication to include dilations


pixel_arrays = []

transformer = Transformer()
visualizer = Visualizer()


def render_animation() -> None:
    for frame_number in range(constants.FRAME_COUNT):
        print(f'\nFrame {frame_number + 1}/{constants.FRAME_COUNT}')

        transformer.step()

        pixel_array = transformer.get_pixel_array()
    
        pixel_arrays.append(pixel_array)

        if frame_number % 10 == 0:
            # visualizer.render_border(transformer.generators[0].border, label=f'border_g0_{frame_number:06d}')
            # visualizer.render_border(transformer.generators[1].border, label=f'border_g1_{frame_number:06d}')
            # visualizer.render_border(transformer.generators[2].border, label=f'border_g2_{frame_number:06d}')

            visualizer.render_frame(pixel_array, label=f'frame_{(frame_number):06d}')

    visualizer.render_animation(pixel_arrays)


def get_time_string(elapsed_seconds: float) -> str:
    hours, seconds = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"


def main():
    start_time = time.time()

    render_animation()

    elapsed_seconds = time.time() - start_time

    print(f'Execution Time: {get_time_string(elapsed_seconds)}')
    

if __name__ == '__main__':
    main()
