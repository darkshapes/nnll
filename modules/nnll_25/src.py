
import re
import os
import hashlib


def extract_tensor_data(source_data_item: dict, id_values: dict) -> dict:
    """
    Extracts shape and key data from the source meta data and put them into id_values\n
    This would extract whatever additional information is needed when a match is found.\n
    :param source_item_data: `dict` Values from the metadata (specifically state dicts layers)
    :param id_values: `dict` Collection of identifiable attributes extracted from the source item
    :return: `dict` Tensor and tensor shape attribute details from the source item
    """
    TENSOR_TOLERANCE = 4e-2  # ? : Move to a config file?
    search_items = ["dtype", "shape"]
    for field_name in search_items:
        if (field_value := source_data_item.get(field_name)) is not None:
            if isinstance(field_value, list):
                field_value = str(field_value)  # Convert shape list to string
            existing_values = id_values.get(field_name, "").split()

            if field_value not in existing_values:  # Add the new value only if it's not already present
                existing_values.append(field_value)
                id_values[field_name] = " ".join(existing_values)

    return {
        "tensors": id_values.get("tensors", 0),
        'shape': id_values.get('shape', None)
    }


def match_pattern_and_regex(reference_data: str, source_item_data: str) -> bool:
    """
    Match a regex pattern to metadata (specifically state dict layers)\n
    :param reference_data: `str` A regex pattern from known identifiers
    :param source_item_data: `str` Values from the metadata (specifically state dicts layers)
    :return: boolean value of match (or not)\n
    note: prep with conditional `if entry.startswith("r'")`
    """
    if source_item_data.startswith("r'"):
        # Regex conversion
        expression = (source_item_data
                      .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                      .replace(".", r"\.")    # Escape literal dots with '\.'
                      .strip("r'")            # Strip the 'r' and quotes from the string
                      )
        print(expression)
        regex_entry = re.compile(expression)
        return bool(regex_entry.search(reference_data))
    elif source_item_data != "" and reference_data != "":
        return source_item_data == reference_data
    else:
        return False


def compute_file_hash(file_name: str) -> str:
    """
    Compute and return the SHA256 hash of a given file.\n
    :param file_name: `str` Valid path to a file
    :return: `str` Hexadecimal representation of the SHA256 hash.
    :raises FileNotFoundError: File does not exist at the specified path.
    :raises PermissionError: Insufficient permissions to read the file.
    :raises IOError:  I/O related errors during file operations.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File '{file_name}' does not exist.")
    else:
        with open(file_name, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
