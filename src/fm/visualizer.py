from datetime import datetime
import os
from typing import List, Tuple

import imageio
import numpy as np

from fm import constants, labels
from fm.transformer import Transformer
from fm.utils import print_percentage


class Visualizer:
    def __init__(self) -> None:
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.directory = f'output/{self.timestamp}'

        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(f'{self.directory}/frames', exist_ok=True)
        os.makedirs(f'{self.directory}/borders', exist_ok=True)


    def render_border(self, border_cells: List[Tuple[float, float]], label: str) -> None:
        pixel_array = np.zeros((constants.BORDER_MAP_SIZE, constants.BORDER_MAP_SIZE), dtype=np.uint8)

        for cell_x, cell_y in border_cells:
            x = int((cell_x + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)
            y = int((cell_y + constants.DOMAIN_RADIUS) / constants.CELL_SIZE)

            symmetric_y = 2 * constants.BORDER_MAP_RADIUS - 1 - y

            pixel_array[x, y] = 255
            pixel_array[x, symmetric_y] = 255

        imageio.imwrite(f'{self.directory}/borders/{label}.png', pixel_array)


    def render_frame(self, pixel_array: np.ndarray, label: str) -> None:
        imageio.imwrite(f'{self.directory}/frames/{label}.png', pixel_array)


    def render_animation(self, pixel_arrays: List[np.ndarray]) -> None:
        filename = f'{self.directory}/fractal_{self.timestamp}.gif'

        imageio.mimsave(
            filename, 
            pixel_arrays, 
            duration=100, 
            loop=0, 
            kwargs={ 'r': 20 }
        )


    def render_debug(self, transformer: Transformer, frame_index: int, pixel_array: np.ndarray) -> None:
        output = 0
        total_outputs = int(constants.DEBUG_FRAME) + transformer.mode.value

        frame_count_digits = len(str(constants.FRAME_COUNT))
        frame_index_output = f'{frame_index:0{frame_count_digits}d}'

        print_percentage(output, total_outputs, labels.DEBUG)

        if constants.DEBUG_FRAME:
            output += 1
            
            self.render_frame(pixel_array, label=f'frame_{frame_index_output}')
            
            print_percentage(output, total_outputs, labels.DEBUG)

        if constants.DEBUG_BORDER:
            for index, generator in enumerate(transformer.generators):
                if generator.active:
                    output += 1

                    self.render_border(generator.border_cells, label=f'border{index}_{frame_index_output}')

                    print_percentage(output, total_outputs, labels.DEBUG)
        
        print_percentage(100, 100, labels.DEBUG)
        print()
        print()


    def print_frame(self, index: int) -> None:
        padding = 12

        frame_text = (
            f'{" " * padding}'
            f'FRAME {(index):0{len(str(constants.FRAME_COUNT))}d}/{constants.FRAME_COUNT}'
            f'{" " * padding}'
        )

        print(f'§{"=" * len(frame_text)}§')
        print(f'§{" " * len(frame_text)}§')
        print(f'§{frame_text}§')
        print(f'§{" " * len(frame_text)}§')
        print(f'§{"=" * len(frame_text)}§')
        
        print()
