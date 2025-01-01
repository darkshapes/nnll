
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from re import L
from modules.nnll_25.src import ExtractAndMatchMetadata

class ValueComparison():
    """
    Loop individual comparisons as situationally required
    """

    def check_inner_values(self, pattern_details: dict, tensor_dimension: dict, tensor_count: int | None = None) -> bool:
        if self.is_identical and 'shapes' not in pattern_details and 'tensors' not in pattern_details:
            return True
        elif self.is_identical and ('shapes' in pattern_details or 'tensors' in pattern_details):
            tensor_dimensions = {}
            if 'shapes' in pattern_details:
                tensor_dimensions['shapes'] = tensor_dimension.get('shape')
            if 'tensors' in pattern_details and tensor_count is not None:
                tensor_dimensions['tensors'] = tensor_count
                if isinstance(pattern_details['tensors'],list): # Can handle multiple tensor length possibilities
                    for count in pattern_details['tensors']: # Pre-check the value, then pop if unnecessary
                        if tensor_dimensions['tensors'] == count:
                            tensor_dimensions.pop('tensors') # Prevents send of both pattern_detal tensor and unpacked tensor
            self.is_identical = all(self.extract.is_pattern_in_layer(pattern_details[k], v) for k, v in tensor_dimensions.items())
        return self.is_identical

    def check_model_identity(self, pattern_details: dict, unpacked_metadata: dict, tensor_count: int | None = None) -> bool:
        """
        Iteratively structure unpacked metadata into reference pattern, then feed a equivalence check.\n
        :param pattern_details: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `bool` Whether or not the values from the model header and tensor_count were found inside pattern_details\n
        The minimum requirement is a **blocks** string value to match, since our shape value is determined by which block is checked
        """
        self.extract = ExtractAndMatchMetadata()
        self.is_identical = False

        for layer_key, tensor_dimension in unpacked_metadata.items():
            if pattern_details and 'blocks' in pattern_details:
                blocks = pattern_details['blocks']
                if not isinstance(blocks, list):
                    blocks = [blocks]

                for block in blocks:

                        self.is_identical = self.extract.is_pattern_in_layer(block, layer_key)
                        #print(layer_key)
                        if self.is_identical == True:
                            check_inner_value = self.check_inner_values(pattern_details, tensor_dimension, tensor_count)
                            if check_inner_value == True:
                                return self.is_identical
            else:
                check_inner_value = self.check_inner_values(pattern_details, tensor_dimension, tensor_count)
                if check_inner_value == True:
                        return self.is_identical

        return self.is_identical



