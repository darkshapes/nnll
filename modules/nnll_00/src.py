

def get_matching_layer(json_file_dict):
    """`
    Test of effectiveness with `map` `lambda` function for nested `dict` traversal
    `json_file_dict` a dictionary of features from recognized models
    """
    matching_layer = next(iter(
        map(lambda stored_value: stored_value, json_file_dict.keys())
    ), 0)
    return matching_layer
