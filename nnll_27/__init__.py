### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def wipe_printer(*formatted_data: dict) -> None:
    """
    Print data sp that it replaces itself in the console buffer\n
    :param formatted_data: Output of `pretty_tabled_output()`
    :return: None
    """
    from sys import stdout

    stdout.write("\033[F\r" * (len(formatted_data)))  # ANSI escape codes to move the cursor up 3 lines
    for line_data in formatted_data:
        stdout.write(" " * 175 + "\x1b[1K\r")
        stdout.write(f"{line_data}\n")  # Print the lines

    stdout.flush()  # Empty output buffer to ensure the changes are shown


def pretty_tabled_output(table_title: str, aggregate_data: dict, width: int = 18) -> None:
    """
    Pretty print data in column format\n
    horizontal divider of arbitrary length
    shrink print columns to data width
    todo: consider shutil to dynamically create
    :param title: `dict` Header key to use for the table
    :param aggregate_data: `dict` A dictionary of values to print
    :return: `dict` A formatted bundle of data ready to print
    """
    table_contents = aggregate_data
    key_value_length = len(table_contents)
    # width_top = key_value_length * 1.5
    info_format = "{:^{width}}|" * key_value_length
    header_keys = tuple(table_contents)
    horizontal_bar = "  " + "-" * (width - 1) * key_value_length
    horizontal_bar += "----"
    formatted_data = tuple(table_contents.values())
    wipe_printer(table_title, info_format.format(*header_keys, width=width), horizontal_bar, info_format.format(*formatted_data, width=width))
