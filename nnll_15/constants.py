### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from sys import modules as sys_modules
from typing import Annotated, Callable, Optional

from pydantic import BaseModel, Field

from nnll_01 import debug_message as dbug
from nnll_01 import debug_monitor
from nnll_01 import info_message as nfo
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

mir_db = JSONCache(CONFIG_PATH_NAMED)

# Order is important for this list since it is used for LibType below
API_NAMES: list = ["ollama", "huggingface_hub", "lmstudio", "vllm", "llamafile", "cortex"]


@debug_monitor
def _check_and_import():
    """Check if the module is available. If not, try to import it dynamically.
    Returns True if the module is successfully imported or already available, False otherwise."""

    for api in API_NAMES:
        try:
            __import__(api)
            # setattr(LibType, api_libtype.get(api), True)
        except ImportError as error_log:
            nfo("|Ignorable| Source unavailable:", f"{api}")
            dbug(error_log)
            # setattr(LibType, api_libtype.get(api), False)


_check_and_import()


class LibType(Enum):
    """API library constants"""

    OLLAMA: int = (0, API_NAMES[0] in sys_modules)
    HUB: int = (1, API_NAMES[1] in sys_modules)
    VLLM: int = (2, API_NAMES[2] in sys_modules)
    LM_STUDIO: int = (3, API_NAMES[3] in sys_modules)
    LLAMAFILE: int = (4, API_NAMES[4] in sys_modules)
    CORTEX: int = (5, API_NAMES[5] in sys_modules)
    # http://localhost:39281
    # LLAMAFILE  : "http://localhost:8080/v1  api_key = "sk-no-key-required"


example_str = ("function_name", "import.function_name")


class GenTypeC(BaseModel):
    """
    Generative inference types in ***C***-dimensional order\n
    ***Compute***, sorted from 'most involved' to 'least involved'\n
    :param clone: Copying identity, voice, exact mirror
    :param sync: Tone, tempo, color, quality, genre, scale, mood
    :param translate: A range of comprehensible approximations
    """

    clone: Annotated[Callable | None, Field(default=None)]
    sync: Annotated[Callable | None, Field(default=None)]
    translate: Annotated[Callable | None, Field(default=None)]


class GenTypeCText(BaseModel):
    """
    Generative inference types in ***C***-dimensional order for text operations\n
    ***Compute***, sorted from 'most involved' to 'least involved'\n
    :param research: Quoting, paraphrasing, and deriving from sources
    :param chain_of_thought: A performance of processing step-by-step
    :param question_answer: Basic, straightforward responses
    """

    research: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    chain_of_thought: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    question_answer: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]


class GenTypeE(BaseModel):
    """
    Generative inference operation types in ***E***-dimensional order \n
    ***Equivalence***, lists sorted from 'highly-similar' to 'loosely correlated.'"\n
    :param universal: Affecting all conversions
    :param text: Text-only conversions
    """

    universal: GenTypeC = GenTypeC(clone=None, sync=None, translate=None)
    text: GenTypeCText = GenTypeCText(research=None, chain_of_thought=None, question_answer=None)


VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "3d render", "vector graphic", "upscale_image"]

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
