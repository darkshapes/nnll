### <!-- // /*  SPDX-License-Identifier: LAL-1.3) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint: disable=import-outside-toplevel

from nnll_01 import debug_monitor


@debug_monitor
def gather_metadata(pattern_reference_path_named: str = None) -> dict:
    """
    Collect data for comparison steps\n
    :param pattern_reference_path_named: `str` Relative or absolute file path and file name of .json filter data
    :return: `dict` The reference data to run the checks
    """
    import os
    from nnll_30 import read_json_file

    if pattern_reference_path_named is None:
        filter_file = "nnll_60"
        pattern_reference_path_named = os.path.join(filter_file, "config", "filter.json")

    pattern_reference = read_json_file(pattern_reference_path_named)
    return pattern_reference


@debug_monitor
def route_metadata(unpacked_metadata: dict, pattern_reference: dict, attributes: int) -> dict:
    """
    Direct metadata information transmission between layers of identity checks\n
    :param pattern_reference: `dict` A dictionary of regex patterns and criteria
    :param unpacked_metadata: `dict` Values from the unknown file metadata (created for state dict layers)
    :param attributes: `dict` Optional additional metadata, such as tensor count and file_size (None will bypass necessity of these matches)
    :return:
    """
    from nnll_46 import IdConductor
    # from nnll_47 import parse_pulled_keys

    conductor_instance = IdConductor()
    layer_keys = conductor_instance.identify_layer_type(pattern_reference, unpacked_metadata, attributes)
    category_type = conductor_instance.identify_category_type(layer_keys, pattern_reference, unpacked_metadata, attributes)
    model_type = conductor_instance.identify_model(category_type, pattern_reference, unpacked_metadata, attributes)
    # pulled_keys = parse_pulled_keys(layer_keys, category_type, model_type)

    # return pulled_keys
