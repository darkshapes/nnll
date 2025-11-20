# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List


from pathlib import Path
from typing import Optional
import os


def ensure_path(folder_path_named: Path, file_name: Optional[str] = None) -> Optional[Path]:
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
