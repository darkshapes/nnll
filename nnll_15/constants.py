### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from typing import Annotated, Callable, Optional

from pydantic import BaseModel, Field
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

mir_db = JSONCache(CONFIG_PATH_NAMED)


class LibType(Enum):
    """API library constants"""

    OLLAMA: int = 0
    HUB: int = 1
    VLLM: int = 2
    LM_STUDIO: int = 3


# z-dimension is graphics hardware / available memory
# see 'nnll_13 sys_cap/spec_writer in 0.1.dev132'


example_str = ("function_name", "import.function_name")


class GenTypeO(BaseModel):
    """
    Generative inference types in O dimensional order\n
    **Operating**, sorted from 'most involved' to 'least involved'\n
    :param clone: Copying identity, voice, exact mirror
    :param sync: Tone, tempo, color, quality, genre, scale, mood
    :param translate: A range of comprehensible approximations
    :param research: Quoting, paraphrasing, and deriving from sources
    :param chain_of_thought: A performance of processing step-by-step
    :param question_answer: Basic, straightforward responses
    """

    clone: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    sync: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    translate: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    research: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    chain_of_thought: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    question_answer: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]


class GenTypeC(BaseModel):
    """
    Generative inference operation types in C-dimensional order \n
    **Converging**, lists sorted from 'highly-similar' to 'loosely correlated.'"\n
    :param universal: Affecting all conversions
    :param text: Text-only conversions
    """

    universal: GenTypeO = GenTypeO()
    text: GenTypeO = GenTypeO()


VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "3d", "upscale_image"]

VALID_TASKS = {
    LibType.OLLAMA: {
        ("image", "text"): ["mllama", "llava", "vllm"],
    },
    LibType.LM_STUDIO: {
        ("image", "text"): [True],
        ("text", "text"): ["llm"],
    },
    LibType.HUB: {
        ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
        ("text", "text"): ["chat", "conversational", "text-generation", "text2text-generation"],
        ("text", "video"): ["video generation"],
        ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
    },
}
