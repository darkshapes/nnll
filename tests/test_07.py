### <!-- // /*  SPDX-License-Identifier: LAL-1.3) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# import unittest

# import os
# import sys

# from nnll_07 import Domain, Architecture, Compatability, Implementation


# class TestDomainArchitectureCompatability(unittest.TestCase):
#     def test_domain_architecture_component(self):
#         # Create a Compatability object
#         component1 = Compatability(model_type="transformer", dtype="float32", file_size=1024, layer_type="huggingface", component_name="bert-base-uncased")

#         # Create another Compatability object
#         component2 = Compatability(model_type="resnet50", dtype="int8", file_size=512, layer_type="torchvision", component_name="resnet50-pretrained")

#         # Create an Architecture object and add components to it
#         architecture1 = Architecture(architecture="nlp")
#         architecture1.add_implementation("transformer", component1)

#         architecture2 = Architecture(architecture="vision")
#         architecture2.add_implementation("resnet50", component2)

#         # Create a Domain object and add architectures to it
#         domain = Domain(domain_name="models")
#         domain.add_architecture("nlp", architecture1)
#         domain.add_architecture("vision", architecture2)

#         # Expected dictionary structure
#         expected_dict = {
#             "models.nlp.transformer": {"model_type": "transformer", "dtype": "float32", "file_size": 1024, "layer_type": "huggingface", "component_name": "bert-base-uncased"},
#             "models.vision.resnet50": {"model_type": "resnet50", "dtype": "int8", "file_size": 512, "layer_type": "torchvision", "component_name": "resnet50-pretrained"},
#         }
#         self.allowed_keys = {"dtype", "file_size", "disk_path", "layer_type", "component_name", "custom_slot_1", "custom_slot_2"}

#         # Convert the domain to a dictionary and assert equality
#         actual_dict = domain.to_dict()
#         self.assertEqual(actual_dict, expected_dict)
