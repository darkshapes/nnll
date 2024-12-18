
from collections import defaultdict
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.pardir, "nnll", "modules")))
from nnll_24.src import ValueComparisons


class BlockScanner:

    def filter_metadata(self, filter_cascade: dict, model_header: dict, tensor_count: int) -> dict:
        """
        Orchestrate navigation through known metadata dictionary to determine an unknown model file's identity\n
        :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        file_metadata = defaultdict(dict)  # A place to store corresponding metadata
        bundle_data = []  # A place to store multiple matching elements
        bundle_types = []

        handle_values = ValueComparisons
        for criteria in filter_cascade:
            # After the first check, if we know it is compvis we have to search all the other
            if file_metadata.get("layer_type") == ["compvis"] and file_metadata.get("category", 0) != 0 and tensor_count > 1100:
                if len(bundle_types) < 1 and criteria == "unet":
                    bundle_check = handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count)
                    next
                else:
                    bundle_check = handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count=None)
                if bundle_check is not None:
                    bundle_data.append(bundle_check)
                    bundle_types.append(criteria)
            elif file_metadata.get("layer_type") is not None and file_metadata.get("category", 0) != 0:
                file_metadata["model"] = handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count)
            else:
                file_metadata[criteria] = handle_values.find_value_path(filter_cascade[criteria], model_header, tensor_count)
                if file_metadata.get("layer_type") is None:
                    file_metadata["layer_type"] = "unknown"
        if bundle_data is not None and bundle_data != []:
            file_metadata["component_name"] = [data for sublist in bundle_data for data in sublist]
            file_metadata["component_type"] = [category for category in bundle_types]
            file_metadata["category"] = "bundle"
            file_metadata["model"] = file_metadata["component_name"][0]
        for element in file_metadata:
            if isinstance(file_metadata[element], list):
                file_metadata[element] = ' '.join(file_metadata[element])

        check_entries = ["category", "model"]
        for entry in check_entries:
            if file_metadata.get(entry) is None:
                file_metadata[entry] = "unknown"
        return file_metadata
