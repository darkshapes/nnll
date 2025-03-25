### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


from nnll_01 import debug_monitor


class LayerFilter:
    """
    Class to direct systematic comparison of model state dict layers
    """

    @debug_monitor
    def __init__(self):
        from nnll_24 import KeyTrail

        self.handle_values = KeyTrail

    @debug_monitor
    def identify_layer_type(self, pattern_reference, unpacked_metadata, tensors):
        bundle_check = self.handle_values.pull_key_names(pattern_reference["layer_type"], unpacked_metadata, tensors)
        layer_type = bundle_check if bundle_check else "unknown"
        return layer_type

    @debug_monitor
    def finalize_metadata(self, file_metadata):
        for key, value in file_metadata.items():
            if isinstance(value, list):
                file_metadata[key] = " ".join(map(str, value))
        return file_metadata

    @debug_monitor
    def reference_walk_conductor(self, pattern_reference: dict, unpacked_metadata: dict, tensors: int) -> dict:
        """
        Navigate through a dictionary of known model attributes to determine an unknown model file's identity\n
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensors: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """

        from collections import defaultdict

        file_metadata = defaultdict(dict)

        file_metadata["layer_type"] = self.identify_layer_type(pattern_reference, unpacked_metadata, tensors)
        bundle_check = file_metadata["layer_type"]
        bundle_types = []  # A place to store multiple matching elements

        if file_metadata.get("layer_type") == "unknown":  # No layer type found
            bundle_types = self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensors)  # Continue anyway
        else:  # When layer type is found
            file_metadata["layer_type"] = bundle_check
            if "modelspec" != file_metadata["layer_type"]:  # only modelspec has bundles
                bundle_types = self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensors)  # Try to find category
            elif tensors is None or tensors < 1100:  # 1100 measured as lowest tensor count for bundled model files
                bundle_types = self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensors)  # Try to find category
            else:
                for category in range(list(pattern_reference["category"].keys())):
                    bundle_types.extend(self.handle_values.pull_key_names(pattern_reference["category"][category], unpacked_metadata, tensors))  # Try to find category

        if bundle_types is None or bundle_types == []:  # If we have no category jump to, search every category
            file_metadata["category"] = "unknown"
            categories = list(pattern_reference.keys())
            for category in categories[2:]:
                file_metadata["model"] = self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensors)

        else:  # When a category is found
            file_metadata["category"] = bundle_types  # Bundle types directs to the relevant categories

            if isinstance(file_metadata["category"], list):  # When more than one value has been found in category
                bundle_check = []  # Empty variable by reinitialization
                for category in file_metadata["category"]:  # Jump to known values inside
                    if len(bundle_check) >= 1 and len(file_metadata["category"]) > 1:  # If we already captured the first model and need more, ignore tensors
                        bundle_check.extend(self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, None))
                    else:
                        bundle_check = self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensors)  # First model only

                if bundle_check is None:  # Nothing matched, assign placeholder
                    file_metadata["model"] = "unknown"

                elif len(bundle_check) > 0 and isinstance(bundle_check, list):  # One or More matches
                    if len(bundle_check) > 1:
                        file_metadata["component_type"] = file_metadata["category"]
                        file_metadata["category"] = "bundle"
                        file_metadata["component_name"] = bundle_check
                    file_metadata["model"] = bundle_check[0]  # Apply the first bundle_check result to model field

            else:  # Category was not a list, only found one
                try:
                    file_metadata["model"] = self.handle_values.pull_key_names(pattern_reference[file_metadata["category"]], unpacked_metadata, tensors)
                except KeyError:  # as error_log:
                    # ("A reference to a key of the filter dictionary failed to be found.")
                    file_metadata["model"] = "unknown"

        return self.finalize_metadata(file_metadata)
