### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


"""為使用者介面清理和安排元資料 Clean and arrange metadata"""

# pylint: disable=line-too-long, import-outside-toplevel

from typing import List, Tuple
from nnll.monitor.file import debug_monitor, dbug, nfo
from nnll.metadata.constants import (
    BracketedDict,
    DownField,
    EmptyField,
    IsThisNode,
    ListOfDelineatedStr,
    NodeNames,
    NodeWorkflow,
    UpField,
)
from nnll.metadata.read_tags import MetadataFileReader


# /______________________________________________________________________________________________________________________ ComfyUI format


@debug_monitor
def clean_with_json(prestructured_data: dict, first_key: str) -> dict:
    """
    Use json loads to arrange half-formatted dict into valid dict\n
    :param prestructured_data: A dict with a single working key
    :param key_name: The single working key name
    :return: The previous data newly formatted as a dictionary, but one key deeper
    """
    import json
    from json import JSONDecodeError

    try:
        cleaned_data = json.loads(prestructured_data[first_key])
    except JSONDecodeError as error_log:
        nfo("Attempted to parse invalid formatting on", prestructured_data, error_log)
        return None
    return cleaned_data


@debug_monitor
def validate_typical(nested_map: dict, key_name: str) -> dict | None:
    """
    Check metadata structure and ensure it meets expectations\n
    :param nested_map: metadata structure
    :type nested_map: dict
    :return: The original node map one key beneath initial entry point, if valid, or None
    """
    from pydantic import ValidationError

    is_search_data = IsThisNode()
    if next(iter(nested_map[key_name])) in NodeWorkflow.__annotations__.keys():
        try:
            is_search_data.workflow.validate_python(nested_map)
        except ValidationError as error_log:  #
            dbug("%s", f"Node workflow not found, returning NoneType {key_name}", error_log)
        else:
            return nested_map[key_name]
    else:
        try:
            is_search_data.data.validate_python(nested_map[key_name])  # Be sure we have the right data
        except ValidationError as error_log:
            dbug("%s", "Node data not found", error_log)
        else:
            return nested_map[key_name]

    dbug("%s", f"Node workflow not found {key_name}")
    dbug(KeyError(f"Unknown format for dictionary {key_name} in {nested_map}"))
    return None


@debug_monitor
def search_for_prompt_in(search_data, accumulated_prompt_data, name_column) -> dict:
    """Check for prompt data in"""
    extracted_data = {}
    data_column_title = NodeNames.DATA_KEYS[name_column]
    prompt_column = search_data[data_column_title]
    for node_field, node_input in prompt_column.items():
        if node_field not in NodeNames.IGNORE_KEYS:
            new_label = node_field
            label_count = str(accumulated_prompt_data.keys()).count(node_field)
            if label_count > 0:
                new_label = f"{node_field}_{label_count}"

            if not isinstance(node_input, list) and node_input:
                if isinstance(node_input, str):
                    node_input = node_input.strip()
                extracted_data.setdefault(new_label, node_input)
    return extracted_data


@debug_monitor
def search_for_gen_data_in(search_data: dict, accumulated_data: dict) -> dict:
    """Check for generative data settings within in a dict structure\n
    Filter based on dict value data type and reference names
    :param search_data: The dict structure to scan
    :param type: `dict`
    :param extracted_gen_data: Collected data from previous scans
    :param type: `dict`
    return: A dict of collected values
    """
    gen_data = {}
    if isinstance(search_data, dict):
        for node_field, node_input in search_data["inputs"].items():
            if node_field not in NodeNames.IGNORE_KEYS:
                new_label = node_field
                label_count = str(accumulated_data.keys()).count(node_field)
                if label_count > 0:
                    new_label = f"{node_field}_{label_count}"
                if not isinstance(node_input, list):
                    if isinstance(node_input, str):
                        node_input = node_input.strip()
                    gen_data.setdefault(new_label, node_input)
    return gen_data


@debug_monitor
def filter_keys_of(normalized_clean_data: dict) -> Tuple[dict]:
    """
    Validate a dictionary structure then scan it based on info type\n
    :param normalized_clean_data: Data prevalidated as correct json dict structure
    :type dict: type
    :return: A pair of data sets corresponding to prompt and generative settings
    """
    extracted_prompt_data = {}
    extracted_gen_data = {}
    if normalized_clean_data is not None:
        for node_number in normalized_clean_data:
            search_data = validate_typical(normalized_clean_data, node_number)  # Returns dictionary of node number
            for name_column_title in NodeNames.DATA_KEYS:
                if search_data and search_data.get(name_column_title, False):
                    name_column = name_column_title
                    node_name = search_data[name_column]
                    break
                node_name = search_data
            if node_name in NodeNames.ENCODERS or node_name in NodeNames.STRING_INPUT:
                extracted_prompt_data.update(search_for_prompt_in(search_data, extracted_prompt_data, name_column))
            else:
                extracted_gen_data.update(search_for_gen_data_in(search_data, extracted_gen_data))
    return extracted_prompt_data, extracted_gen_data


@debug_monitor
def redivide_nodeui_data_in(header: str, first_key: str) -> Tuple[dict]:
    """
    Orchestrate tasks to recreate dictionary structure and extract relevant keys within\n
    :param header: Embedded dictionary structure
    :type variable: str
    :param section_titles: Key names for relevant data segments
    :type variable: list
    :return: Metadata dict, or empty dicts if not found
    """

    sorted_header_prompt = {}
    sorted_header_data = {}
    try:
        jsonified_header = clean_with_json(header, first_key)
    except KeyError as error_log:
        nfo("%s", "No key found.", error_log)
        return {"": EmptyField.PLACEHOLDER, " ": EmptyField.PLACEHOLDER}
    else:
        if first_key == "workflow":
            normalized_clean_data = {"1": jsonified_header}  # To match normalized_prompt_structure format
        else:
            normalized_clean_data = jsonified_header
        sorted_header_prompt, sorted_header_data = filter_keys_of(normalized_clean_data)

        return sorted_header_prompt, sorted_header_data


@debug_monitor
def arrange_nodeui_metadata(header_data: dict) -> dict:
    """
    Using the header from a file, run formatting and parsing processes \n
    Return format : {"Prompts": , "Settings": , "System": } \n
    :param header_data: Header data from a file
    :return: Metadata in a standardized format
    """

    extracted_prompt_pairs, generation_data_pairs = redivide_nodeui_data_in(header_data, "prompt")
    if extracted_prompt_pairs == {}:
        gen_pairs_copy = generation_data_pairs.copy()
        extracted_prompt_pairs, second_gen_map = redivide_nodeui_data_in(header_data, "workflow")
        generation_data_pairs = second_gen_map | gen_pairs_copy
    return {
        UpField.PROMPT: extracted_prompt_pairs or {EmptyField.PLACEHOLDER: EmptyField.EMPTY},
        DownField.GENERATION_DATA: generation_data_pairs or {EmptyField.PLACEHOLDER: EmptyField.EMPTY},
    }


# /______________________________________________________________________________________________________________________ A4 format


@debug_monitor
def delineate_by_esc_codes(text_chunks: dict, extra_delineation: str = "'Negative prompt'") -> list:
    """
    Format text from header file by escape-code delineations\n
    :param text_chunk: Data from a file header
    :return: text data in a dict structure
    """

    segments = []
    replace_strings = ["\xe2\x80\x8b", "\x00", "\u200b", "\n", extra_delineation]
    dirty_string = text_chunks.get("parameters", text_chunks.get("exif", False))  # Try parameters, then "exif"
    if not dirty_string:
        return []  # If still None, then just return an empty list

    if isinstance(dirty_string, bytes):  # Check if it is bytes
        try:
            dirty_string = dirty_string.decode("utf-8")  # If it is, decode into utf-8 format
        except UnicodeDecodeError as error_log:  # Catch exceptions if they fail to decode
            nfo("Failed to decode", dirty_string, error_log)
            return []  # Return nothing if decoding fails

    segments = [dirty_string]  # All string operations after this should be safe

    for buffer in replace_strings:
        new_segments = []
        for delination in segments:  # Split segments and all sub-segments of this string
            split_segs = delination.split(buffer)
            new_segments.extend(s for s in split_segs if s)  # Skip empty strings
        segments = new_segments

    clean_segments = segments
    return clean_segments


@debug_monitor
def extract_prompts(clean_segments: list) -> Tuple[dict, str]:
    """
    Split string by pre-delineated tag information\n
    :param cleaned_text: Text without escape codes
    :type cleaned_text: list
    :return: A dictionary of prompts and a str of metadata
    """

    if len(clean_segments) <= 2:
        prompt_metadata = {
            "Positive prompt": clean_segments[0],
            "Negative prompt": "",
        }
        deprompted_text = " ".join(clean_segments[1:])
    elif len(clean_segments) > 2:
        prompt_metadata = {
            "Positive prompt": clean_segments[0],
            "Negative prompt": clean_segments[1].strip("Negative prompt':"),
        }
        deprompted_text = " ".join(clean_segments[2:])
    else:
        return None
    return prompt_metadata, deprompted_text


@debug_monitor
def validate_mapping_bracket_pair_structure_of(possibly_invalid: str) -> str:
    """
    Take a string and prepare it for a conversion to a dict map\n
    :param possibly_invalid: The string to prepare
    :type possibly_invalid: `str`
    :return: A correctly formatted string ready for json.loads/dict conversion operations
    """

    _, kv_pairs = possibly_invalid
    bracketed_kv_pairs = BracketedDict(bracketed=kv_pairs)

    # There may also need to be a check for quotes here

    valid_kv_pairs = BracketedDict.model_validate(bracketed_kv_pairs)
    # Removes pydantic annotations
    pair_string = str(valid_kv_pairs.bracketed).replace("'", '"')
    return pair_string


@debug_monitor
def extract_dict_by_delineation(deprompted_text: str) -> Tuple[dict, list]:
    """
    Split a string with partial dictionary delineations into a dict\n
    :param cleaned_text: Text without escape codes
    :return: A freeform string and a partially-organized list of metadata
    """

    import re

    @debug_monitor
    def repair_flat_dict(traces_of_pairs: List[str]) -> dict:
        """
        Convert a list with the first element as a key into a string with delinations \n
        :param traces_of_kv_pairs: that has a string with dictionary delineations as its second value\n
        Examples - List[str]`['Key', '{Key, Value}'] , ['Key:', ' "Key": "Value" '] `\n
        NOT `"[Key, Key, Value]"` or other improper dicts\n
        :type traces_of_kv_pairs: `list`
        :return: A corrected dictionary structure from the kv pairs
        """

        delineated_str = ListOfDelineatedStr(convert=traces_of_pairs)
        key, _ = next(iter(traces_of_pairs))
        validated_string = validate_mapping_bracket_pair_structure_of(delineated_str.convert)
        repaired_sub_dict[key] = validated_string
        return repaired_sub_dict

    key_value_trace_patterns = [
        # r"\s*([^:,]+):\s*([^,]+)",
        r" (\w*): ([{].*[}]),",
        r' (\w*\s*\w+): (["].*["]),',
    ]
    repaired_sub_dict = {}
    remaining_text = deprompted_text

    # Main loop, run through regex patterns
    for pattern in key_value_trace_patterns:  # if this is a dictionary
        traces_of_pairs = re.findall(pattern, remaining_text)  # Search matching text in the original string
        if traces_of_pairs:
            repair_flat_dict(traces_of_pairs)
            remaining_text = re.sub(pattern, "", remaining_text, 1)  # Remove matching text in the original string
        # Handle for no match on generation as well

        else:  # Safely assume this is not a dictionary
            return {}, remaining_text
    return repaired_sub_dict, remaining_text


@debug_monitor
def make_paired_str_dict(text_to_convert: str) -> dict:
    """
    Convert an unstructured metadata string into a dictionary\n
    :param dehashed_data: Metadata tags with quote and bracket delineated features removed
    :return: A valid dictionary structure with identical information
    """

    segmented = text_to_convert.split(", ")
    delineated = [item.split(": ") for item in segmented if isinstance(item, str) and ": " in item]
    try:
        converted_text = {el[0]: el[1] for el in delineated if len(el) == 2}
    except IndexError as error_log:
        dbug("Index position for prompt input out of range", text_to_convert, "at", delineated, error_log, tb=error_log.__traceback__)
        converted_text = None
    return converted_text


@debug_monitor
def arrange_webui_metadata(header_data: str) -> dict:
    """
    Using the header from a file, send to multiple formatting, cleaning, and parsing, processes \n
    Return format : {"Prompts": , "Settings": , "System": }\n
    :param header_data: Header data from a file
    :return: Metadata in a standardized format
    """

    cleaned_text = delineate_by_esc_codes(header_data)
    prompt_map, deprompted_text = extract_prompts(cleaned_text)
    system_map, generation_text = extract_dict_by_delineation(deprompted_text)
    generation_map = make_paired_str_dict(generation_text)
    return {
        UpField.PROMPT: prompt_map,
        DownField.GENERATION_DATA: generation_map,
        DownField.SYSTEM: system_map,
    }


# /______________________________________________________________________________________________________________________ EXIF Tags


@debug_monitor
def arrange_exif_metadata(header_data: dict) -> dict:
    """Arrange EXIF data into correct format"""
    for tag, dirty_string in header_data.items():
        if isinstance(dirty_string, bytes):  # Check if it is bytes
            decoded_data = dirty_string.decode("utf-16-be")  # If it is, decode into utf-16 format
        else:
            decoded_data = header_data[tag]
        # print(decoded_data)
        if isinstance(decoded_data, str):
            map_start = decoded_data.find("{")
            decoded_string = decoded_data.replace("'", '"')
            scrub_dictionary = {tag: decoded_string[map_start:]}
            sorted_header_prompt, sorted_header_data = redivide_nodeui_data_in(scrub_dictionary, tag)
            if sorted_header_prompt and sorted_header_prompt != {}:
                return {UpField.METADATA: sorted_header_prompt, DownField.GENERATION_DATA: sorted_header_data}
            else:
                return {
                    UpField.TAGS: {DownField.EXIF: {key: value} for key, value in sorted_header_data.items() if (key != "icc_profile" or key != "exif")},
                    DownField.ICC: {UpField.DATA: sorted_header_data.get("icc_profile")},
                    DownField.EXIF: {
                        UpField.DATA: sorted_header_data.get("exif"),
                    },
                }


# /______________________________________________________________________________________________________________________ Module Interface


@debug_monitor
def coordinate_metadata_operations(header_data: dict | str, metadata: dict = None) -> dict:
    """
    Process data based on identifying contents\n
    :param header_data: Metadata extracted from file
    :type header_data: dict
    :param datatype: The kind of variable storing the metadata
    :type datatype: str
    :param metadata: The filtered output extracted from header data
    :type metadata: dict
    :return: A dict of the metadata inside header data
    """

    if isinstance(header_data, dict):
        metadata = arrange_dict_metadata(header_data)
    elif isinstance(header_data, str):
        metadata = arrange_str_metadata(header_data)
    if not metadata:
        nfo("Failed to find/load metadata : %s", header_data)
        metadata = {EmptyField.PLACEHOLDER: {EmptyField.PLACEHOLDER: EmptyField.EMPTY}}

    return metadata


def arrange_dict_metadata(header_data: dict) -> dict:
    """
    Perform simple processing of a dictionary based on expectation of metadata\n
    :param header_data: The data from the file
    :type header_data: str
    :return: A dictionary formatted for display
    """
    metadata = {}
    condition_functions = {
        lambda: header_data.get("prompt"): arrange_nodeui_metadata,
        lambda: header_data.get("parameters"): arrange_webui_metadata,
        lambda: header_data.get("UserComment"): arrange_exif_metadata,
    }
    schema_functions = {
        UpField.METADATA,
        DownField.LAYER_DATA,
        UpField.TEXT_DATA,
        DownField.JSON_DATA,
        DownField.TOML_DATA,
    }

    for condition, arranger in condition_functions.items():
        if condition():
            metadata.update(arranger(header_data))
            break
    for field_title in schema_functions:
        field_data = header_data.get(field_title)
        if field_data:
            if field_title in metadata:
                field_title_count = str(metadata.keys()).count(field_data)
                field_title_name = f"{field_title}_{field_title_count}"
                metadata.setdefault(field_title_name, field_data)
            else:
                metadata.setdefault(field_title, field_data)

    return metadata


@debug_monitor
def arrange_str_metadata(header_data: str) -> dict:
    """
    Perform simple processing of raw text strings inside of a file\n
    :param header_data: The data from the file
    :type header_data: str
    :return: A dictionary formatted for display
    """
    from json import JSONDecodeError

    metadata = {}
    try:
        metadata = dict(header_data)
    except JSONDecodeError as error_log:
        dbug("JSON Decode failed %s", error_log, tb=error_log.__traceback__)
        metadata = {UpField.DATA: header_data, EmptyField.PLACEHOLDER: EmptyField.EMPTY}
    except KeyError as error_log:
        dbug("Could not load metadata as dict %s", error_log, tb=error_log.__traceback__)
        metadata = {UpField.DATA: header_data, EmptyField.PLACEHOLDER: EmptyField.EMPTY}
    else:
        up_data = {UpField.DATA: header_data}
        down_data = {EmptyField.PLACEHOLDER: EmptyField.EMPTY}
        metadata.setdefault(UpField.DATA, up_data)
        metadata.setdefault(EmptyField.PLACEHOLDER, down_data)

    return metadata


@debug_monitor
def parse_metadata(file_path_named: str) -> dict:
    """
    Extract the metadata from the header of an image file\n
    :param file_path_named: The file to open
    :return: The metadata from the header of the file
    """

    reader = MetadataFileReader()
    metadata = {}
    header_data = reader.read_header(file_path_named)
    if header_data is not None:
        metadata = coordinate_metadata_operations(header_data)
        if metadata == {EmptyField.PLACEHOLDER: {"": EmptyField.PLACEHOLDER}} or not isinstance(metadata, dict):
            nfo("Unexpected format", file_path_named)
    return metadata
