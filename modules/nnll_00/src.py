
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


def deepest_key_of(nested_dictionary):
    """
    Nnested `dict` value retrieval\n
    :param nested_dictionary: `dict` The nested dictionary to parse
    :return: `tuple` The keys leading to the deepest value.
    """
    matching_layer = next(iter(
        map(lambda value_contents: value_contents, nested_dictionary.keys())
    ), 0)
    return matching_layer
