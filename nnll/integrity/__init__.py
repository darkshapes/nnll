#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path
from typing import Optional
import os


def ensure_path(folder_path_named: Path, file_name: Optional[str] = None) -> Optional[Path]:
    """Provide absolute certainty a file location exists\n
    :param folder_path_named: Location to test, defaults to os.path.dirname(HOME_FOLDER_PATH)
    :param file_name:Optional file name to test, defaults to None
    :return: The original folder path, or none if failure
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
