from typing import Union


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

            result.append("" + char.lower()) if index != 0 else result.append(char.lower())
        else:
            result.append(char)
    return "".join(result).lower()
