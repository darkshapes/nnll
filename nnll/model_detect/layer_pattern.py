### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from math import isclose
import re

from nnll.monitor.file import debug_monitor


class ExtractAndMatchMetadata:
    @debug_monitor
    def is_pattern_in_layer(self, block_pattern: list | str, layer_element: list) -> bool:
        """
        Match a string, int or regex pattern to metadata (specifically state dict layers)\n
        :param block_pattern: `str` | `int` Regex patterns, strings, or number from known identifiers
        :param layer_element: `list` Values from the metadata as str or int
        :return: boolean value of match (or not)\n
        note: prep with conditional `if entry.startswith("r'")`
        """
        if layer_element == "" or layer_element is None or block_pattern is None:
            return False
        if isinstance(layer_element, str):
            if layer_element.startswith("r'"):
                # Regex conversion
                expression_pattern = (
                    layer_element.replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                    .replace(".", r"\.")  # Escape literal dots with '\.'
                    .strip("r'")  # Strip the 'r' and quotes from the string
                )
                # print(expression)
                in_parsed_layer = re.compile(expression_pattern)
                return bool(in_parsed_layer.search(block_pattern))
            else:
                return block_pattern in layer_element
        elif isinstance(block_pattern, list) and isinstance(layer_element, list):  # This will never be from 'blocks'
            return block_pattern == layer_element
        elif isinstance(block_pattern, int) and isinstance(layer_element, int):
            if block_pattern == layer_element or isclose(block_pattern, layer_element, rel_tol=1e-1):
                return True
        return False
