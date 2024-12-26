
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from modules.nnll_25.src import ExtractAndMatchMetadata

class ValueComparison:
    """
    Loop individual comparisons as situationally required
    """

    @classmethod
    def compare_values(cls, nested_attributes: dict, model_header: dict, tensor_count: int | None = None) -> bool:
        """
        Iterate through a model metadata in order to compare pattern against model_header.
        If "blocks" is a list, iterate through it, checking both regex and str pattern matches.\n
        :param nested_filter: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `bool` Whether or not the values from the model header and tensor_count were found inside nested_filter
        """
        extract = ExtractAndMatchMetadata()

        if nested_attributes is not None or nested_attributes != {}:
            # Iterate through the filter, run the scan
            # Determine what needs to be done with the data, then execute
            if len(nested_attributes) != 0 and nested_attributes.get("blocks") is not None:
                for layer, tensor_data in model_header.items():  # repeat for every incoming layer
                    if isinstance(nested_attributes["blocks"], list) == True:
                        for pattern in nested_attributes["blocks"]:  # repeat for each list item
                            cls.match = extract.match_pattern_and_regex(pattern, layer)
                            if cls.match == True:
                                break

                    else:
                        cls.match = extract.match_pattern_and_regex(nested_attributes["blocks"], layer)

                    if hasattr(cls, "match") :
                        if cls.match == True:
                            model_dict = {}
                            # Fetch any remaining items necessary to compare
                            if nested_attributes.get("tensors", None) is None and nested_attributes.get("shapes", None) is None:
                                return True
                            if nested_attributes.get("shapes", None) is not None:
                                model_dict.setdefault("shapes", tensor_data.get("shape", 0))
                            if nested_attributes.get("tensors", None) is not None and tensor_count is not None:
                                model_dict.setdefault("tensors", tensor_count)
                            elif tensor_count is None and nested_attributes.get("tensors", None) is not None:
                                # if 'tensor_count was not supplied, ignore it
                                nested_attributes.pop("tensors")

                            # Do a quick match of all values at once
                            if all(extract.match_pattern_and_regex(nested_attributes.get(key), value) for key, value in model_dict.items()):
                                return True
        return False
