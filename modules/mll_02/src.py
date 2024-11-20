# mock_data = {
#     "neuralnet":
#     {
#         "type_1": 1,
#         "type_2": 2
#     },
#     "type_1": {
#         "model_name":
#         {
#             "layers": "model.diffusion_model.attn.norm1.weight",
#             "tensors": 3,
#             "shape": {
#                 768, 8
#             },
#             "hash": "0b0c3b4a0"
#         }
#     }
# }


# mock_data = {
#     "neuralnet":
#     {
#         "type_1": 1,
#         "type_2": 2
#     },
#     "type_1": {
#         "model_name": {
#             "model.diffusion_model.attn.norm1.weight",
#             3,
#             {768, 8},
#             "0b0c3b4a0"
#         }
#     }
# }


# def get_all_keys(mock_data: dict):
#     for key, value in mock_data.items():
#         yield key
#         if isinstance(value, dict):
#             yield from get_all_keys(value)


# # for x in get_all_keys(mock_data):
# #     print(x)

# from itertools import chain
# data = chain.from_iterable(mock_data)
# map(lambda stored_value: stored_value, data)

mock_data = {
    "neuralnet": {
        "type_1": 1,
        "type_2": 2
    },
    "type_2": {
        "model_name": {
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
    },
    "type_1": {
        "model_name": {
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
    }
}

def find_matching_model_recursive(file_attributes, data_dict):
    """
    Recursive method to crawl dictionary for key matches
    `file attributes` examination data of file attributes
    `data_dict` precollected and identified attributes
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            result = find_matching_model_recursive(file_attributes, value)
            if result:
                return result
        else:
            if all(attr in file_attributes.values() for attr in file_attributes.values()):
                # Return both True and the matching model name.
                return True, key
    return False, None

# Test the recursive approach
file_attributes = {
    "layers": "model.diffusion_model.attn.norm1.weight",
    "tensors": 3,
    "shape": {768, 8},
    "hash": "0b0c3b4a0"
}

# Should return (True, 'model_name')
print(find_matching_model_recursive(file_attributes, mock_data))
