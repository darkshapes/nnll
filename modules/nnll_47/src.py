#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s
# def parse_pulled_keys(layer_keys: dict, category_keys: str | list, model_keys: list) -> dict:


def parse_pulled_keys(layer_keys: dict, category_keys: str | list, model_keys: list) -> dict:
    """
    Clean up identity keys for later steps\n
    :param layer_keys: `dict` Identified model layer type
    :param category_keys: `list` Identified model category
    :param model_keys: `list` Identified model types
    :return: `dict` Keysc
    """
    pulled_keys = layer_keys.copy()

    if isinstance(category_keys, str):
        category_keys = [category_keys]

    if isinstance(model_keys, str):
        model_keys = [model_keys]

    if model_keys is not None:
        pulled_keys['architecture'] = next(iter(key for key in model_keys if key is not None), 'unknown')
    if category_keys is not None:
        pulled_keys['model_type'] = next(iter(key for key in category_keys if key is not None),'unknown')
    #for key in [*category_keys, *model_keys]:
    pulled_keys['component_type'] = ''.join(map(str, ['unknown' if x is None else x for x in category_keys]))
    pulled_keys['component_name'] = ''.join(map(str, ['unknown' if x is None else x for x in model_keys]))

    # Join lists, replace `None` with 'unknown'
    for key, value in pulled_keys.items():
        if isinstance(value, list):
            pulled_keys[key] = ''.join(['unknown' if v is None else str(v) for v in value])

    return pulled_keys
