### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


from nnll.monitor.file import debug_monitor


class ValueComparison:
    """
    Loop individual comparisons as situationally required
    """

    @debug_monitor
    def __init__(self):
        from nnll.model_detect.layer_pattern import ExtractAndMatchMetadata

        self.extract = ExtractAndMatchMetadata()
        self.is_identical = False

    @debug_monitor
    def check_inner_values(self, pattern_details: dict, tensor_dimension: dict, attributes: dict | None = None) -> bool:
        from nnll.integrity.hashing import compute_hash_for

        possible_attributes = ["tensors", "file_size", "shapes", "hash"]
        if self.is_identical and not any(key in pattern_details for key in possible_attributes):
            return True
        elif self.is_identical and any(key in pattern_details for key in possible_attributes):
            tensor_dimensions = {}
            if "shapes" in pattern_details:
                tensor_dimensions["shapes"] = tensor_dimension.get("shape")

            for attribute_detail in possible_attributes[:2]:  # Only want the first two values in possible attributes
                if attribute_detail in pattern_details and attributes is not None and attributes.get(attribute_detail, None) is not None:
                    tensor_dimensions[attribute_detail] = attributes[attribute_detail]
                    if isinstance(pattern_details[attribute_detail], list):  # Can handle multiple tensor length possibilities
                        for count in pattern_details[attribute_detail]:  # Pre-check the value, then pop if unnecessary
                            if tensor_dimensions[attribute_detail] == count:
                                tensor_dimensions.pop(attribute_detail)  # Prevents send of both pattern_detal tensor and unpacked tensor

            if "hash" in pattern_details and attributes.get("file_path_named", None) is not None:
                tensor_dimensions["hash"] = compute_hash_for(attributes["file_path_named"])
            self.is_identical = all(self.extract.is_pattern_in_layer(pattern_details[k], v) for k, v in tensor_dimensions.items())
        return self.is_identical

    @debug_monitor
    def check_model_identity(self, pattern_details: dict, unpacked_metadata: dict, attributes: dict | None = None) -> bool:
        """
        Iteratively structure unpacked metadata into reference pattern, then feed a equivalence check.\n
        :param pattern_details: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata
        :param attributes: `dict` Optional additional metadata, such as tensor count and file_size (None will bypass necessity of these matches)
        :return: `bool` Whether or not the values from the model header and tensors were found inside pattern_details\n
        The minimum requirement is a **blocks** string value to match, since our shape value is determined by which block is checked
        """
        for layer_key, tensor_dimension in unpacked_metadata.items():
            if pattern_details and "blocks" in pattern_details:
                blocks = pattern_details["blocks"]
                if not isinstance(blocks, list):
                    blocks = [blocks]

                for block in blocks:
                    self.is_identical = self.extract.is_pattern_in_layer(block, layer_key)
                    # print(layer_key)
                    if self.is_identical:
                        check_inner_value = self.check_inner_values(pattern_details, tensor_dimension, attributes)
                        if check_inner_value:
                            return self.is_identical
            else:
                check_inner_value = self.check_inner_values(pattern_details, tensor_dimension, attributes)
                if check_inner_value:
                    return self.is_identical

        return self.is_identical
