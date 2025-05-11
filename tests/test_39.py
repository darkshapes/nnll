### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# import unittest
# from unittest.mock import Mock, patch, MagicMock

# from nnll_39 import route_metadata
# # from nnll_51 import parse_pulled_keys


# class TestRouteMEtadata(unittest.TestCase):
#     @patch("nnll_39.IdConductor")
#     # @patch("nnll_39.parse_pulled_keys")
#     def test_route_metadata(self, mockPullKeys, mockConductor):
#         conductor = mockConductor()
#         conductor.identify_layer_type = MagicMock()
#         conductor.identify_category_type = MagicMock()
#         conductor.identify_model = MagicMock()
#         conductor.identify_layer_type.return_value = {"data_type": "無價值的"}  # Worthless
#         conductor.identify_category_type.return_value = {"component_type": "無價值的"}
#         conductor.identify_model.return_value = {"無價值的"}
#         result = route_metadata({"ref": 1}, {"layer_type": {"blocks": 1}, "category": {"blocks": "x"}}, 5)

#         conductor.identify_layer_type.assert_called()
#         conductor.identify_category_type.assert_called()
#         conductor.identify_model.assert_called()
#         assert result == mockPullKeys()
