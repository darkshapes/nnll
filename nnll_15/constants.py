### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from sys import modules as sys_modules
from typing import Annotated, Callable, Optional, Tuple

from pydantic import BaseModel, Field

from nnll_01 import debug_message as dbug
from nnll_01 import debug_monitor
from nnll_01 import info_message as nfo
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

mir_db = JSONCache(CONFIG_PATH_NAMED)

# Order is important for this list since it is used for LibType below
#                   0                1              2           3           4       5
API_NAMES: list = ["ollama", "huggingface_hub", "lmstudio", "cortex", "llamafile", "vllm"]


@debug_monitor
def check_and_import() -> Tuple[bool]:
    """Check if the module is available. If not, try to import it dynamically.
    Returns True if the module is successfully imported or already available, False otherwise."""

    import requests
    import httpcore
    import json
    from urllib3.exceptions import NewConnectionError, MaxRetryError

    cortex_server: bool = False
    llamafile_server: bool = False
    for api in API_NAMES:
        if api == "cortex":
            try:
                response = requests.get("http://127.0.0.1:39281/v1/chat/completions", timeout=(3, 3))
                if response is not None:
                    data = response.json()
                    if data.get("result") == "OK":
                        cortex_server = True
            except (
                httpcore.ConnectError,
                json.decoder.JSONDecodeError,
                requests.exceptions.ConnectionError,
                ConnectionRefusedError,
                MaxRetryError,
                NewConnectionError,
                TimeoutError,
                OSError,
            ):
                continue

        elif api == "llamafile":
            try:
                import openai
            except (ModuleNotFoundError, ImportError) as error_log:
                dbug(error_log)
                continue
            else:
                try:
                    response = requests.get("http://localhost:8080/v1", timeout=(3, 3))
                    if response is not None:
                        data = response.json()
                        if data.get("result") == "OK":
                            llamafile_server = True
                except (
                    openai.APIConnectionError,
                    httpcore.ConnectError,
                    json.decoder.JSONDecodeError,
                    requests.exceptions.ConnectionError,
                    ConnectionRefusedError,
                    MaxRetryError,
                    NewConnectionError,
                    TimeoutError,
                    OSError,
                ):
                    continue
        else:
            try:
                __import__(api)
                # setattr(LibType, api_libtype.get(api), True)
            except ImportError as error_log:
                nfo("|Ignorable| Source unavailable:", f"{api}")
                dbug(error_log)
                # setattr(LibType, api_libtype.get(api), False)
    return cortex_server, llamafile_server


cortex_server, llamafile_server = check_and_import()


class LibType(Enum):
    """API library constants"""

    OLLAMA: int = (0, API_NAMES[0] in sys_modules)
    HUB: int = (1, API_NAMES[1] in sys_modules)
    LM_STUDIO: int = (2, API_NAMES[2] in sys_modules)
    CORTEX: int = (3, cortex_server)
    LLAMAFILE: int = (4, llamafile_server)
    VLLM: int = (5, API_NAMES[5] in sys_modules)


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
    LibType.CORTEX: {
        ("text", "text"): ["text"],
    },
    LibType.OLLAMA: {
        ("text", "text"): ["mllama", "llava", "vllm"],
    },
    LibType.LLAMAFILE: {
        ("text", "text"): ["text"],
    },
    LibType.LM_STUDIO: {
        ("image", "text"): [("vision", True)],
        ("text", "text"): ["llm", ("tool", True)],
    },
    LibType.HUB: {
        ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
        ("text", "text"): ["chat", "conversational", "text-generation", "text2text-generation"],
        ("text", "video"): ["video generation"],
        ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
    },
}
