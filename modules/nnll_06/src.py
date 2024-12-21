#// SPDX-License-Identifier: MIT
#// d a r k s h a p e s


def compare_dicts(known, metadata):
    """
    Simple dictionary match crawler prototype
    `known` dict of attributes for identified models
    `metadata` detected attributes of model to identify
    """
    for k, v in known.items():
        if isinstance(v, dict):
            if not compare_dicts(v, metadata.get(k, {})):
                return False
        elif metadata.get(k) != v:
            return False
    return True
