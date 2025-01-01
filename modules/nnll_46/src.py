
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s
#
from modules.nnll_24.src import KeyTrail
from collections import defaultdict

class IdConductor:
    """Navigate through a dictionary of known model attributes to determine an unknown model file's identity\n"""

    def identify_model(self, category_type, pattern_reference, unpacked_metadata, tensor_count=None):
        """
        Operate model id search functions\n
        :param category_type: `dict` Relevant values to identify the unknown model identity
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        key_trail = KeyTrail()

        self.model_keys = []
        if isinstance(category_type, str):
            category_type = [category_type]

        for index, key_name in enumerate(category_type):
            expression = (pattern_reference[key_name], unpacked_metadata,)
            if index == 0 :
                expression += (tensor_count,)
            self.model_keys.append(key_trail.pull_key_names(*expression))

        return self.model_keys

    def identify_category_type(self, layer_keys: dict, pattern_reference:dict, unpacked_metadata:dict, tensor_count=None) -> dict:
        """
        Operate category type search functions\n
        :param layer_keys: `dict` Relevant values to identify the unknown model category
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        key_trail = KeyTrail()
        self.category_keys = []
        self.category_keys.append(key_trail.pull_key_names(pattern_reference["category"], unpacked_metadata, tensor_count))

        if layer_keys['layer_type'] == 'compvis' and tensor_count > 1100: # Compvis UNet model attributes
            for category in list(pattern_reference["category"])[3:]: # Ignore TAESD, LoRA (irrelevant) and UNet (already been checked)
                self.category_keys.append(key_trail.pull_key_names(pattern_reference["category"][category], unpacked_metadata))
        return self.category_keys

    def identify_layer_type(self, pattern_reference, unpacked_metadata, tensor_count=None):
        """
        Operate model layer search functions\n
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria known to identify models
        :param unpacked_metadata: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `dict` A mapping of attributes identifying the unknown model file and its constituent parts
        """
        key_trail = KeyTrail()
        self.layer_keys = defaultdict(dict)
        self.layer_keys["layer_type"] = key_trail.pull_key_names(pattern_reference["layer_type"], unpacked_metadata, tensor_count)
        return self.layer_keys
