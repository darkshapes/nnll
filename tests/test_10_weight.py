# import unittest
# from unittest.mock import MagicMock

# # from nnll_15 import RegistryEntry


# class TestIntent(unittest.TestCase):
#     def setUp(self):
#         """Initialize necessary objects for testing"""
#         from nnll_10 import IntentProcessor

#         self.ip = IntentProcessor()
#         self.ip.intent_graph = MagicMock()

#     def test_edit_weight_zero_base_weight(self):
#         """Test case when base_weight is zero"""
#         selection = "test_selection"
#         index_num = 1
#         self.ip.intent_graph.edges.return_value = [("node1", "node2", {"entry": MagicMock(model="test_model"), "entry": MagicMock(model="test_model")})]

#         self.ip.edit_weight(
#             selection,
#             base_weight=1.0,
#         )

#         # Assertions to check if weight was incremented by 0.1
#         self.assertEqual(self.ip.intent_graph["node1"]["node2"][index_num]["weight"], 0.1)

#     # def test_edit_weight_non_zero_base_weight(self):
#     #     """Test case when base_weight is non-zero"""
#     #     selection = "test_selection"
#     #     index_num = 0
#     #     self.ip.intent_graph.edges.return_value = [("node1", "node2", {"entry": MagicMock(model="test_model")})]
#     #     base_weight = 5.0
#     #     self.ip.edit_weight(selection, base_weight=base_weight, index_num=index_num)

#     #     # Assertions to check if weight was decremented by 0.1

#     #     self.assertEqual(self.ip.intent_graph["node1"]["node2"][index_num]["weight"], base_weight - 0.1)


# if __name__ == "__main_":
#     unittest.main()
