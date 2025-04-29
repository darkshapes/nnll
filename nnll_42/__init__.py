### <!-- // /*  SPDX-License-Identifier: LAL-1.3) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def populate_module_index(active_directories: list, indicator: list) -> dict:
    """
    Generate a page of absolute path links to the catalog modules, including function names\n
    Gather function names in the module as determined by keywords \n
    Set format as markup link - '[nnll_## - function_names, etc_etc](link_to_absolute_path)'\n
    Add or append the value to the index\n
    :param active_directories: `list` The folders to search for modules within
    :param indicator: `list` Names of relevant module files
    :return: `dict` of all present modules
    """
    import os
    import pathlib
    from collections import defaultdict
    import re

    module_index = defaultdict(dict)
    for folder_location in active_directories:
        for file_name in indicator:
            input_file = os.path.join(folder_location, file_name)

            if not os.path.exists(input_file):
                next
            else:
                cat_numbers = []
                catalog_number = pathlib.Path(folder_location).parts
                source_code = open(input_file, encoding="UTF-8")

                object_name = [re.search(r"\b(?:def|class)\s+(?P<title>\w+)", line) for line in source_code]  # if "__init__" not in line and "main" not in line
                if object_name:
                    key = f"[{catalog_number[-1]} - {', '.join({obj.group('title') for obj in object_name if obj is not None})}]"

                    if len(key) > 12 and not module_index.get(key):  # [nnll_xx - ] <- 12 chars, skip empty, I know, im lazy here, w/e
                        value = f"{module_index[key]} {input_file}"
                        module_index.setdefault(key, value)
                        cat_numbers.append(catalog_number[-1])

    return module_index
