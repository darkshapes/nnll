### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from typing import Iterable, List, Union


def slice_number(text: str) -> Union[int, float, str]:
    """Separate a numeral value appended to a string\n
    :return: Converted value as int or float, or unmodified string
    """
    for index, char in enumerate(text):  # Traverse forwards
        if char.isdigit():
            numbers = text[index:]
            if "." in numbers:
                return float(numbers)
            try:
                return int(numbers)
            except ValueError:
                return numbers
    return text


def snake_caseify(camel_case: str, delimiter: str = "_") -> str:
    """
    Turn mixed case string with potential acronyms into delimiter-separated string.\n
    :param camel_case: Incoming camel-cased or acronym-containing string
    :return: A string with capitals separated by the delimiter, preserving acronyms 🐍🐛
    """
    result = []
    acronyms = ["ldm3d"]
    ignore_phrases = ["DiT", "AuraFlow", "LDM", "HunyuanVideo", "AnimateDiff"]
    for acro in acronyms:
        camel_case = camel_case.replace(acro[1:], f"{acro[1:]}".lower())
    for phrase in ignore_phrases:
        camel_case = camel_case.replace(phrase, f"{phrase}".lower())
    for index, char in enumerate(camel_case):
        if char.isupper():
            if index > 0 and camel_case[index - 1].islower():  # Detect change of cases
                result.append(delimiter)

            if index > 0 and camel_case[index - 1].isupper():  # Did case change? Don't adjust
                result.append(char.lower())
                continue

            result.append("" + char.lower()) if index != 0 else result.append(char.lower())  # pylint:disable=expression-not-assigned
        else:
            result.append(char)
    return "".join(result).lower()


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
    user_input = input(f"{polite_msg} {preposition} {tag}: ")
    if not user_input and not required:
        return None
    input_store.append(user_input)
    while True:
        if user_input and input_store:
            metadata = f"{more} {preposition}"
            user_input = input(f"{polite_msg} {metadata} {tag} (leave blank to skip): ")
            if user_input:
                input_store.append(user_input)
            else:
                return input_store
