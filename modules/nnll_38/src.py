#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import re
import os
import hashlib
import sys

class ExtractValueData:

    def extract_tensor_data(self, source_data_item: dict, id_values: dict) -> dict:
        """
        Extracts shape and key data from the source meta data and put them into id_values\n
        This would extract whatever additional information is needed when a match is found.\n
        :param layer_element: `dict` Values from the metadata (specifically state dicts layers)
        :param id_values: `dict` Collection of identifiable attributes extracted from the source item
        :return: `dict` Tensor and tensor shape attribute details from the source item
        """
        TENSOR_TOLERANCE = 4e-2  # ? : Move to a config file?
        search_items = ["dtype", "shape"]
        for field_name in search_items:
            if (field_value := source_data_item.get(field_name)) is not None:
                if isinstance(field_value, list):
                    field_value = str(field_value)  # Convert shape list to string
                existing_values = id_values.get(field_name, "").split()

                if field_value not in existing_values:  # Add the new value only if it's not already present
                    existing_values.append(field_value)
                    id_values[field_name] = " ".join(existing_values)

        return {
            "tensors": id_values.get("tensors", 0),
            'shape': id_values.get('shape', None)
        }
