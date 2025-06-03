### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from typing import Tuple, Callable
from rich.color import Color


def grey_nouveau_theme():
    """Set a a midnight blue and dark purple theme to console logs"""
    from rich import style
    from rich.theme import Theme

    greyscale = spectrum(18, 16, 16)  # pylint-unnecessary-lambda-assignment # noqa

    NOUVEAU: Theme = Theme(
        {
            "log.message": style.Style(color="cyan", dim=True),  # parenthesis # variable name, normal strings
            "repr.ellipsis": style.Style(color=greyscale[1]),
            "log.time": style.Style(color=greyscale[2], dim=True),
            "log.path": style.Style(color=greyscale[1], dim=True),  # line number
            "repr.str": style.Style(color=greyscale[4], dim=True),
            "repr.tag_start": style.Style(color=greyscale[8]),  # class name tag
            "logging.level.notset": style.Style(dim=True),  # level ids
            "log.level": style.Style(color=greyscale[9], dim=True),
            "logging.level.debug": style.Style(color=greyscale[10]),
            "json.str": style.Style(color=greyscale[12], italic=False, bold=False),
            "logging.level.warning": style.Style(color=greyscale[12]),
            "logging.keyword": style.Style(bold=True, color=greyscale[12], dim=True),
            "repr.tag_end": style.Style(color=greyscale[12]),  # class name tag
            "repr.tag_contents": style.Style(color=greyscale[10]),  # class readout
            "logging.level.error": style.Style(color=greyscale[13], bold=True, reverse=True),
            "logging.level.critical": style.Style(color=greyscale[14], bold=True, reverse=True),
            "logging.level.info": style.Style(color=greyscale[15]),
        }
    )
    return NOUVEAU


def divide_spectrum(red_slice: int = None, green_slice: int = None, blue_slice: int = None, function: Callable = Color.from_rgb) -> Tuple[int, int, int]:
    ceiling_num = 255.0
    slice_range = [abs(red_slice), abs(green_slice), abs(blue_slice)]
    max_slice = max(1, *slice_range)  # most repetitions
    rgb_256 = []
    color_steps = []
    for color in slice_range:
        color_steps.append((ceiling_num / color))
    interval_counter = [1, 1, 1]
    for step in range(1, max_slice + 1):
        for i, color in enumerate(slice_range):
            if max_slice * step >= color_steps[i] * interval_counter[i]:
                interval_counter[i] += 1
        r_val = min(int(ceiling_num / slice_range[0] * interval_counter[0]), 255)
        g_val = min(int(ceiling_num / slice_range[1] * interval_counter[1]), 255)
        b_val = min(int(ceiling_num / slice_range[2] * interval_counter[2]), 255)
        rgb_256.append(function(r_val, g_val, b_val))
    return rgb_256


spectrum: Tuple[int, int, int] = lambda red_255, green_255, blue_255: dict(
    enumerate(divide_spectrum(red_255, green_255, blue_255)),
)  # pylint-unnecessary-lambda-assignment
