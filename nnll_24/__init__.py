### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
import sys

from nnll_33 import ValueComparison


class KeyTrail:
    """
    Class for route tracing nested dictionary keys
    """

    @classmethod
    def pull_key_names(cls, pattern_reference: dict, unpacked_metadata: dict, attributes: int | None = None) -> list | None:
        """
        Instantiate comparison function and run a quick check on top-level criteria, otherwise run recursion function\n
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria
        :param unpacked_metadata: `dict` Values from the unknown file metadata (created for state dict layers)
        :param attributes: `dict` Optional additional metadata, such as tensor count and file_size (None will bypass necessity of these matches)
        :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
        """
        compare = ValueComparison()

        def sink_into(next_pattern_reference: dict, flat_key_trail: list = []) -> list | None:
            """
            Recurse through dictionary and return parent keys on boolean condition\n
            :param next_pattern_reference: `dict` A sub-key from `pattern_reference`
            :param flat_key_trail: `list` Sub-keys of `pattern_reference` leading to the current
            :return: `list` The path of keys through the target `dict` if true, or None.
            """
            for current_key, pattern_details in next_pattern_reference.items():
                flat_key_trail.append(current_key)

                if isinstance(pattern_details, dict):  # Check if we've reached the bottom
                    if compare.check_model_identity(pattern_details, unpacked_metadata, attributes) == True:
                        return flat_key_trail[-1]  # Return last found key only (as list)

                    detected_key = sink_into(pattern_details, flat_key_trail)  # Recurse into deeper levels
                    if detected_key is not None:
                        return detected_key  # Return path once found
                else:
                    continue  # Skip non-dict values

            return None  # No matching subtree found

        return sink_into(pattern_reference)  # Begin recursion loop
