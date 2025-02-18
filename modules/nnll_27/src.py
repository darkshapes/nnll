# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

import sys


def wipe_printer(*formatted_data: dict) -> None:
    """
    Print data sp that it replaces itself in the console buffer\n
    :param formatted_data: Output of `pretty_tabled_output()`
    :return: None
    """
    sys.stdout.write("\033[F\r" * (len(formatted_data)))  # ANSI escape codes to move the cursor up 3 lines
    for line_data in formatted_data:
        sys.stdout.write(" " * 175 + "\x1b[1K\r")
        sys.stdout.write(f"{line_data}\n")  # Print the lines

    sys.stdout.flush()  # Empty output buffer to ensure the changes are shown


def pretty_tabled_output(title: str, aggregate_data: dict) -> dict:
    """
    Pretty print data in column format\n
    :param title: `str` Header Value to use for the table
    :param aggregate_data: `dict` A dictionary of values to print
    :return: `dict` A formatted bundle of data ready to print
    """
    print_title = {}
    print_title.setdefault("category", title)
    print_values = print_title | aggregate_data.copy()
    truncate_keys = ["architecture", "file_name", "disk path"]
    for key in truncate_keys:
        if print_values.get(key) is not None:
            print_values.pop(key)  # Only pop if a valid key is found
    key_value_length = len(print_values)  # number of items detected in the scan
    # width_top = key_value_length * 1.5
    width = 18
    info_format = "{:^{width}}|" * key_value_length  # shrink print columns to data width
    header_keys = tuple(print_values)  # use to create table
    horizontal_bar = "  " + "-" * (width - 1) * key_value_length  # horizontal divider of arbitrary length. could use shutil to dynamically create but eh. already overkill
    formatted_data = tuple(print_values.values())  # data extracted from the scan
    wipe_printer(title, info_format.format(*header_keys, width=width), horizontal_bar, info_format.format(*formatted_data, width=width))  # send to print function
