### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def capture_title_numeral(file_name: str, query: str) -> int:
    """
    Retrieve a number from a file name string using the characters that directly follow it\n
    :param filename: `str` The string name of the file (with or without path)
    :param query: `str` String of characters following the number to search for
    :return: `int` Integer of the number value found in the string
    """
    import re
    import os

    try:
        query = query.lower()  # normalize string formatting; digits are focus
        lowercase_filename = os.path.basename(file_name.lower())  # enforce file name only input
        index_of_match = lowercase_filename.rindex(query)  # find match
    except ValueError as error_log:
        print(f"File not named with search_term {error_log}.")
    else:
        if index_of_match is not None:  # found something
            preceding_number = str(lowercase_filename[index_of_match - 2 : index_of_match])  # digits preceding to trim
            preceding_number = re.sub(r"^\W+|\W+$", "", preceding_number)  # no character once or more ending with no character, once or more
            if preceding_number.isdigit():
                captured_number = int(preceding_number)  # double digit number
                return captured_number
            elif preceding_number[1:].isdigit():
                captured_number = int(preceding_number[1:])  # single digit num
                return captured_number
