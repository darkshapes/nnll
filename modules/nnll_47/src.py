#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s
def parse_pulled_keys(self, layer_keys: dict, category_keys: list, model_keys: list) -> dict:
    """
    Clean up identity keys for later steps\n
    :param layer_keys: `dict` Identified model layer type
    :param category_keys: `list` Identified model category
    :param model_keys: `list` Identified model types
    :return: `dict` Keysc
    """
    pulled_keys = layer_keys
    for key in category_keys:
        pulled_keys.setdefault(
            'component_type',
            pulled_keys.get('component_type',[]).append(key)
        )
    pulled_keys['model_type'] = category_keys[:1]
    for key in model_keys:
        pulled_keys.setdefault(
            'component_name',
            pulled_keys.get('component_name',[]).append(key)
        )
    pulled_keys['architecture'] = model_keys[:1]
    for key, value in pulled_keys.items():
        if isinstance(value, list):
            pulled_keys[key] = ' '.join(map(str, value))
        elif value is None:
            pulled_keys[key] = 'unknown'

    return pulled_keys


