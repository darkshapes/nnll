#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import pytest
from modules.nnll_38.src import ExtractValueData

# NOTE: doesnt do anything yet, function was abandoned

testdata_00 = [     # Test basic functionality
    ({"dtype": "float32", "shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({"dtype": "int64", "shape": [5, 5]}, {"dtype": "float32"}, {"tensors": 0, "shape": "[5, 5]", "dtype": "float32 int64"}),


]

testdata_01 = [    # Test with existing values in id_values
    ({"dtype": "float32", "shape": [10, 20]}, {"dtype": "float32", "shape": "[5, 5]"}, {"tensors": 0, "shape": "[5, 5] [10, 20]", "dtype": "float32"}),
    ({"dtype": "int64", "shape": [10, 20]}, {"dtype": "float32 int64", "shape": "[5, 5]"}, {"tensors": 0, "shape": "[5, 5] [10, 20]", "dtype": "float32 int64"}),
]

testdata_02 = [    # Test with list and string conversion
    ({"dtype": "float32"}, {}, {"tensors": 0, "shape": None}),
    ({"shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({}, {"dtype": "float32", "shape": "[5, 5]"}, {"tensors": 0, "shape": "[5, 5]", "dtype": "float32"}),
]
testdata_03 = [    # Test with missing fields in source_data_item
    ({"shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({"shape": [1, 2, 3]}, {"shape": "[10, 20]"}, {"tensors": 0, "shape": "[10, 20] [1, 2, 3]"}),
]
testdata_04 = [  # Test with empty source_data_item and id_values
    ({}, {}, {"tensors": 0, "shape": None}),
]
testdata_05 = [
    ({"dtype": "float32", "shape": 10}, {}, {"tensors": 0, "shape": "10"}),
]
testdata_06 = [
    ({"dtype": None, "shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({"dtype": "float32", "shape": None}, {}, {"tensors": 0, "shape": None}),
]

combined_testdata = testdata_00 + testdata_01 + testdata_02 + testdata_03 + testdata_04 + testdata_05 + testdata_06

class TensorTest:

    @ pytest.mark.parametrize("source_data_item, id_values, expected", )
    def __init__(source_data_item, id_values, expected):
        test_module = ExtractValueData()
        assert test_module.extract_tensor_data(source_data_item, id_values) == expected


