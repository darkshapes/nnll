
from collections import defaultdict
import sys
import os

from modules.nnll_24.src import ValuePath

class BlockScanner():

    handle_values = ValuePath

    def filter_metadata(self, filter_cascade: dict, model_header: dict, tensor_count: int) -> dict:
        """
        Navigate through a dictionary of known model attributes to determine an unknown model file's identity\n
        :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        file_metadata = defaultdict(dict)  # A place to store corresponding metadata
        bundle_data = []  # A place to store multiple matching elements
        bundle_types = []
        bundle_check = self.handle_values.find_value_path(filter_cascade["layer_type"], model_header, tensor_count)
        if bundle_check is not None:
            bundle_data.append(bundle_check)
            file_metadata["layer_type"] = bundle_check
        bundle_types.append(self.handle_values.find_value_path(filter_cascade["category"], model_header, tensor_count))
        if len(bundle_types) > 0:
            bundle_data.append(bundle_types[0])
            file_metadata["category"] = bundle_types[0]
            if file_metadata["category"] == "unknown":
                file_metadata["model"] = "unknown"
                return file_metadata
            elif isinstance(file_metadata["category"], list):
                bundle_check = []
                for category in file_metadata["category"]:
                    if len(bundle_check) >= 1 and len( file_metadata["category"] ) > 1:
                        bundle_check.extend(self.handle_values.find_value_path(filter_cascade[category], model_header, tensor_count=None))
                    else:
                        bundle_check = (self.handle_values.find_value_path(filter_cascade[category], model_header, tensor_count))
                if bundle_check is None:
                    file_metadata["model"] = "unknown"
                elif len(bundle_check) > 0 and isinstance(bundle_check,list):
                    if len(bundle_check) > 1:
                        file_metadata["component_type"] = file_metadata["category"]
                        file_metadata["category"] = "bundle"
                        file_metadata["component_name"] = bundle_check
                    file_metadata["model"] = bundle_check[0]
            else:
                try:
                    file_metadata["model"] = self.handle_values.find_value_path(filter_cascade[file_metadata["category"]], model_header, tensor_count)
                except KeyError as error_log:
                    #print("A reference to a key of the filter dictionary failed to be found.")
                    file_metadata["category"] = "unknown"
                    file_metadata["model"] = "unknown"
        else:
            file_metadata["category"] = self.handle_values.find_value_path(filter_cascade["category"], model_header, tensor_count)
            if file_metadata["category"] is not None:
                file_metadata["model"] = self.handle_values.find_value_path(filter_cascade[file_metadata["category"]], model_header, tensor_count)
            else:
                categories = filter_cascade.keys()
                for category in categories[2:]:
                    file_metadata["model"] = self.handle_values.find_value_path(filter_cascade[category], model_header, tensor_count)
        for element in file_metadata:
            if isinstance(file_metadata[element], list):
                file_metadata[element] = ' '.join(file_metadata[element])
        return file_metadata


        # for criteria in filter_cascade:
        #     # After the first check, if we know it is compvis we have to search all the other categories


        #     if file_metadata.get("layer_type") == ["compvis"] and file_metadata.get("category", 0) == 0 and tensor_count > 1100:
        #         if len(bundle_types) < 1 and criteria == "unet": #if we dont have one category yet
        #             bundle_check = self.handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count)
        #             next
        #         else:
        #             bundle_check = self.handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count=None)
        #         if bundle_check is not None:
        #             bundle_data.append(bundle_check)
        #             bundle_types.append(criteria)
        #     elif file_metadata.get("layer_type") is not None and file_metadata.get("category", 0) != 0:
        #         file_metadata["model"] = self.handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count)
        #     else:
        #         file_metadata[criteria] = self.handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count)
        #         if file_metadata[criteria] is None:
        #             file_metadata[criteria] = "unknown"
        # if bundle_data is not None and bundle_data != []:
        #     file_metadata["component_name"] = [data for sublist in bundle_data for data in sublist]
        #     file_metadata["component_type"] = [category for category in bundle_types]
        #     file_metadata["category"] = "bundle"
        #     file_metadata["model"] = file_metadata["component_name"][0]
        # for element in file_metadata:
        #     if isinstance(file_metadata[element], list):
        #         file_metadata[element] = ' '.join(file_metadata[element])

        # check_entries = ["category", "model"]
        # for entry in check_entries:
        #     if file_metadata.get(entry) is None:
        #         file_metadata[entry] = "unknown"
        return file_metadata
