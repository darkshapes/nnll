# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List

from pathlib import Path
import os


def ensure_path(folder_path_named: Path, file_name: str | None = None) -> Path | None:
    """Provide absolute certainty a file location exists\n
    :param folder_path_named: Location to test
    :param file_name: Optional file name to test, defaults to None
    :return: The original folder path, the original folder path with file, or None if failure
    """
    folder_path_named = Path(folder_path_named)
    if not folder_path_named.exists():
        try:
            folder_path_named.mkdir(parents=True, exist_ok=True)  # Ensure the directory is created resiliently
        except OSError:
            try:
                os.makedirs(folder_path_named, exist_ok=False)
            except OSError:
                return None

    if file_name:
        full_path = os.path.join(folder_path_named, file_name)
        full_path = Path(full_path)
        if full_path.exists():
            return full_path
        try:
            full_path.touch(exist_ok=False)  # Create the file only if it doesn't exist
        except (FileExistsError, OSError):
            pass
        return str(full_path)

    return str(folder_path_named) if folder_path_named.exists() else None


def ask_multi_input(
    tag: str,
    polite_msg: str = "Please provide",
    preposition: str = "metadata for",
    more: str = "additional",
    required: bool = True,
) -> List[str]:
    """Looping `input` to create metadata survey lists of user input under a single label\n
    :param tag: A label for the incoming metadata
    :param polite_msg: Introduction prefix, defaults to "Please provide"
    :param preposition: Partial sentence following the message, defaults to "metadata for"
    "param more: Statement to append for repeated prompts
    :param required: Whethr the field MUST be answered, defaults to True
    :return: A list of answers from the user
    """
    input_store = []
    for prompt in [polite_msg, preposition]:
        prompt = prompt.strip()
    user_input = None
    while True:
        if user_input and input_store:
            metadata = f"{more} {preposition}"
            user_input = input(f"{polite_msg} {metadata} {tag} (leave blank to skip): ")
            if user_input:
                input_store.append(user_input)
            else:
                return input_store
        elif not user_input and not required:
            return None
        else:
            user_input = input(f"{polite_msg} {preposition} {tag}: ")
            input_store.append(user_input)


def prefix_inner_caps(text: str) -> str:
    import re

    return re.sub(r"(?<!^)([A-Z])(?!$)", r"_\1", text)


def generate_valid_resolutions(initial_width: int, initial_height: int) -> list[tuple[int, int]]:
    """Generate valid resolutions based on initial width/height using patch calculations.\n
    :param initial_width: Initial image width
    :param initial_height: Initial image height
    :returns: List of valid (width, height) tuples sorted by aspect ratio"""
    import math

    height_patches = math.ceil(initial_height / 16)
    width_patches = math.ceil(initial_width / 16)
    total_patches = height_patches * width_patches

    valid_resolutions = []

    for h_patches in range(1, total_patches + 1):
        if total_patches % h_patches == 0:
            w_patches = total_patches // h_patches
            height_max = 16 * h_patches
            width_max = 16 * w_patches
            if height_max <= 16383 and width_max <= 16383:  # max WebP pixels
                valid_resolutions.append((width_max, height_max))

    valid_resolutions.sort(key=lambda x: x[0] / x[1] if x[1] > 0 else 0)

    return valid_resolutions


def check_optional_import(module_name: str) -> tuple[bool, any]:
    """Check if an optional module can be imported.\n
    :return: A tuple of the module's availability and the module itself"""
    try:
        module = __import__(module_name)
        return True, module
    except ImportError:
        return False, None
