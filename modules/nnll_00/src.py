#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


def get_matching_layer(json_file_dict):
    """
    Test of effectiveness with `map` `lambda` function for nested `dict` path retrieval\n
    :param json_file_dict: `dict` The nested dictionary to parse
    :return: `tuple` The keys leading to the deepest value.
    """
    matching_layer = next(iter(
        map(lambda stored_value: stored_value, json_file_dict.keys())
    ), 0)
    return matching_layer
