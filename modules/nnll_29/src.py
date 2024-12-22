#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from collections import defaultdict
import sys
import os

from modules.nnll_24.src import ValuePath

class LayerFilter:
    """
    Class to direct systematic comparison of model state dicts
    """

    handle_values = ValuePath

    def filter_metadata(self, filter_cascade: dict, model_header: dict, tensor_count: int) -> dict:
        """
        Navigate through a dictionary of known model attributes to determine an unknown model file's identity\n
        :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        file_metadata = defaultdict(dict)  # A place to store corresponding metadata
        bundle_types = [] # A place to store multiple matching elements

        bundle_check = self.handle_values.find_value_path(filter_cascade["layer_type"], model_header, tensor_count) # Try to find layer type

        if bundle_check is None: # No layer type found
            file_metadata["layer_type"] = "unknown"
            bundle_types = self.handle_values.find_value_path(filter_cascade["category"], model_header, tensor_count) # Continue anyway
        else: # When layer type is found
            file_metadata["layer_type"] = bundle_check
            if "compvis" != file_metadata["layer_type"]: # only compvis has bundles
                bundle_types = self.handle_values.find_value_path(filter_cascade["category"], model_header, tensor_count) # Try to find category
            elif tensor_count is None or tensor_count < 1100: # 1100 measured as lowest tensor count for bundled model files
                bundle_types = self.handle_values.find_value_path(filter_cascade["category"], model_header, tensor_count) # Try to find category
            else:
                for category in range(list(filter_cascade["category"].keys())):
                    bundle_types.extend(self.handle_values.find_value_path(filter_cascade["category"][category], model_header, tensor_count)) # Try to find category

        if bundle_types is None or bundle_types == []: # If we have no category jump to, search every category
            file_metadata["category"] = "unknown"
            categories = list(filter_cascade.keys())
            for category in categories[2:]:
                file_metadata["model"] = self.handle_values.find_value_path(filter_cascade[category], model_header, tensor_count)

        else: # When a category is found
            file_metadata["category"] = bundle_types # Bundle types directs to the relevant categories

            if isinstance(file_metadata["category"], list): # When more than one value has been found in category
                bundle_check = [] # Empty variable by reinitialization
                for category in file_metadata["category"]: # Jump to known values inside
                    if len(bundle_check) >= 1 and len(file_metadata["category"]) > 1: # If we already captured the first model and need more, ignore tensors
                        bundle_check.extend(self.handle_values.find_value_path(filter_cascade[category], model_header, tensor_count=None))
                    else:
                        bundle_check = (self.handle_values.find_value_path(filter_cascade[category], model_header, tensor_count)) # First model only

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
                    file_metadata["model"] = self.handle_values.find_value_path(filter_cascade[file_metadata["category"]], model_header, tensor_count)
                except KeyError as error_log:
                    #("A reference to a key of the filter dictionary failed to be found.")
                    file_metadata["model"] = "unknown"

        for element in file_metadata: # collect all the information into strings (maybe should be done closer to print statement?)
            if isinstance(file_metadata[element], list):
                file_metadata[element] = ' '.join(file_metadata[element])
        return file_metadata