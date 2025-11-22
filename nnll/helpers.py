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


def soft_random(size: int = 0x100000000) -> int:  # previously 0x2540BE3FF
    """Generate a deterministic random number using philox\n
    :params size: `int` RNG ceiling in hex format
    :returns: `int` a random number of the specified length\n
    pair with `random.seed()` for best effect"""

    import secrets

    from numpy.random import Generator, Philox, SeedSequence

    entropy = f"0x{secrets.randbits(128):x}"  # good entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    return int(rndmc.integers(0, size))


def hard_random(hardness: int = 5) -> int:
    """Generate a cryptographically secure random number\n
    :param hardness: `int` byte length of generated number
    :returns: `int` Non-prng random number"""
    from secrets import token_hex

    return int(token_hex(hardness), 16)


def seed_planter(seed: int = soft_random(), deterministic: bool = False, device: str = "cpu") -> int:
    """Force seed number to all available devices\n
    :param seed: The number to grow all random generation from, defaults to `soft_random` function
    :param deterministic: Identical number provides identical output, defaults to True
    :param device: Processor to use, defaults to `first_available(assign=False)` function
    :return: The `int` seed that was provided to the functions."""

    from numpy import random
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if "mps" in device:
        torch.mps.manual_seed(seed)

    return seed


def random_tensor_from_gpu(device: str, input_seed: int = soft_random()):
    """Create a random tensor shape (for testing or other purposes)\n
    :params device: `str` device to assign generation to, defaults to `first_available()` function
    :params input_seed: `int` the seed to control randomization, defaults to soft_random() generator
    :returns: `tensor` Random dimensional tensor"""

    import torch

    torch.set_num_threads(1)
    if input_seed is not None:
        seed_planter(device=device)
    return torch.rand(1, device=device)


def random_int_from_gpu(input_seed: int = soft_random()) -> int:
    """Generate a random number via pytorch
    :params input_seed: `int`a seed to feed the random generator, defaults to `soft_random()` function
    :returns: `int` A random number from current device"""
    import torch

    torch.set_num_threads(1)
    return torch.random.seed() if input_seed is None else torch.random.manual_seed(input_seed)
