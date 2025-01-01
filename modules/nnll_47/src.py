#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s
def parse_pulled_keys(layer_keys: dict, category_keys: str | list, model_keys: list) -> dict:
    """
    Clean up identity keys for later steps\n
    :param layer_keys: `dict` Identified model layer type
    :param category_keys: `list` Identified model category
    :param model_keys: `list` Identified model types
    :return: `dict` Keysc
    """
    pulled_keys = layer_keys
    if isinstance(category_keys, str):
        category_keys = [category_keys]
    for key in category_keys:
        if key is None:
                pulled_keys['component_type'] = 'unknown'
        else:
            pulled_keys['component_type'] = pulled_keys.get('component_type','').join(map(str, key))

    pulled_keys['model_type'] = category_keys[:1]
    for key in model_keys:
        if key is None:
                pulled_keys['component_name'] = 'unknown'
        else:
            pulled_keys['component_name'] = pulled_keys.get('component_name','').join(map(str, key))

    pulled_keys['architecture'] = model_keys[:1]
    for key, value in pulled_keys.items():
        if isinstance(value, list):
            pulled_keys[key] = ' '.join(map(str, value))
        elif value is None:
            pulled_keys[key] = 'unknown'

    return pulled_keys


