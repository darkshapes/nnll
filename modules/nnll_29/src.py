
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from collections import defaultdict

from modules.nnll_24.src import KeyTrail

class LayerFilter:
    """
    Class to direct systematic comparison of model state dict layers
    """

    def identify_layer_type(self, pattern_reference, unpacked_metadata, tensor_count):
        bundle_check = self.handle_values.pull_key_names(pattern_reference["layer_type"], unpacked_metadata, tensor_count)
        layer_type = bundle_check if bundle_check else "unknown"
        return layer_type

    # def identify_category_types(self, pattern_reference, unpacked_metadata, tensor_count, layer_type):
    #     if layer_type == "compvis" and (tensor_count is None or tensor_count < 1100):
    #         return [self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensor_count)]
    #     elif layer_type != "unknown":
    #         bundle_types = []
    #         for category in pattern_reference["category"].values():
    #             bundle_types.append(self.handle_values.pull_key_names(category, unpacked_metadata, tensor_count))
    #         return bundle_types
    #     return []

    # def identify_models(self, pattern_reference, unpacked_metadata, tensor_count, category_type):
    #     if not category_type:
    #         categories = list(pattern_reference.keys())[2:]
    #         model_type = "unknown"
    #         for category in categories:
    #             model_type = self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensor_count)
    #             if model_type:
    #                 break
    #     else:
    #         model_type = self.identify_bundled_models(pattern_reference, unpacked_metadata, tensor_count, category_type)
    #     return model_type

    # def identify_bundled_models(self, pattern_reference, unpacked_metadata, tensor_count, category_type):
    #     model_types = []
    #     for category in category_type:
    #         if len(model_types) >= 1 and len(category_type) > 1:
    #             model_types.append(self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata))
    #         else:
    #             model_types = self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensor_count)
    #     return model_types if model_types else "unknown"

    def finalize_metadata(self, file_metadata):
        for key, value in file_metadata.items():
            if isinstance(value, list):
                file_metadata[key] = ' '.join(map(str, value))
        return file_metadata

    def reference_walk_conductor(self, pattern_reference: dict, unpacked_metadata: dict, tensor_count: int) -> dict:
        """
        Navigate through a dictionary of known model attributes to determine an unknown model file's identity\n
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        file_metadata = defaultdict(dict)
        self.handle_values = KeyTrail
        file_metadata["layer_type"] = self.identify_layer_type(pattern_reference, unpacked_metadata, tensor_count)
        bundle_check = file_metadata["layer_type"]

        # file_metadata["category_type"] = self.identify_category_types(pattern_reference, unpacked_metadata, tensor_count, file_metadata["layer_type"])
        # file_metadata["model_type"] = self.identify_models(pattern_reference['category'], unpacked_metadata, tensor_count, file_metadata["category_type"])


        # file_metadata['component_type'] = str(
        #     file_metadata["category_type"] if isinstance(file_metadata["category_type"], str)
        #     else ' '.join(map(str, file_metadata["category_type"]))
        #     )

        # file_metadata['component_name'] = file_metadata["model_type"]

        # return self.finalize_metadata(file_metadata)
        file_metadata = defaultdict(dict)  # A place to store corresponding metadata
        bundle_types = [] # A place to store multiple matching elements

        # bundle_check = self.handle_values.pull_key_names(pattern_reference["layer_type"], unpacked_metadata, tensor_count) # Try to find layer type

        if file_metadata.get("layer_type") == "unknown": # No layer type found
            bundle_types = self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensor_count) # Continue anyway
        else: # When layer type is found
            file_metadata["layer_type"] = bundle_check
            if "compvis" != file_metadata["layer_type"]: # only compvis has bundles
                bundle_types = self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensor_count) # Try to find category
            elif tensor_count is None or tensor_count < 1100: # 1100 measured as lowest tensor count for bundled model files
                bundle_types = self.handle_values.pull_key_names(pattern_reference["category"], unpacked_metadata, tensor_count) # Try to find category
            else:
                for category in range(list(pattern_reference["category"].keys())):
                    bundle_types.extend(self.handle_values.pull_key_names(pattern_reference["category"][category], unpacked_metadata, tensor_count)) # Try to find category

        if bundle_types is None or bundle_types == []: # If we have no category jump to, search every category
            file_metadata["category"] = "unknown"
            categories = list(pattern_reference.keys())
            for category in categories[2:]:
                file_metadata["model"] = self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensor_count)

        else: # When a category is found
            file_metadata["category"] = bundle_types # Bundle types directs to the relevant categories

            if isinstance(file_metadata["category"], list): # When more than one value has been found in category
                bundle_check = [] # Empty variable by reinitialization
                for category in file_metadata["category"]: # Jump to known values inside
                    if len(bundle_check) >= 1 and len(file_metadata["category"]) > 1: # If we already captured the first model and need more, ignore tensors
                        bundle_check.extend(self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensor_count=None))
                    else:
                        bundle_check = (self.handle_values.pull_key_names(pattern_reference[category], unpacked_metadata, tensor_count)) # First model only

                if bundle_check is None: # Nothing matched, assign placeholder
                    file_metadata["model"] = "unknown"

                elif len(bundle_check) > 0 and isinstance(bundle_check,list): # One or More matches
                    if len(bundle_check) > 1:
                        file_metadata["component_type"] = file_metadata["category"]
                        file_metadata["category"] = "bundle"
                        file_metadata["component_name"] = bundle_check
                    file_metadata["model"] = bundle_check[0] # Apply the first bundle_check result to model field


            else: # Category was not a list, only found one
                try:
                    file_metadata["model"] = self.handle_values.pull_key_names(pattern_reference[file_metadata["category"]], unpacked_metadata, tensor_count)
                except KeyError as error_log:
                    #("A reference to a key of the filter dictionary failed to be found.")
                    file_metadata["model"] = "unknown"

        return self.finalize_metadata(file_metadata)

