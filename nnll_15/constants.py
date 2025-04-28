### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from typing import Annotated, Callable, Optional

from pydantic import BaseModel, Field

from nnll_01 import debug_message as dbug
from nnll_01 import debug_monitor
from nnll_01 import info_message as nfo

@debug_monitor
def has_api(api_name: str) -> bool:
    """Check available modules, try to import dynamically.
    True for successful import, else False."""

    import requests
    import httpcore
    import json
    from urllib3.exceptions import NewConnectionError, MaxRetryError
    if api_name == "lmstudio":
        try:
            from lmstudio import LMStudioWebsocketError
        except (ImportError, ModuleNotFoundError) as error_log:
            nfo("|Ignorable| Source unavailable:", f"{api_name}")
            dbug(error_log)
        else:
            try:
                response = requests.get("http://localhost:1234/v1", timeout=(1, 1))
                data     = response.json() if response is not None else {}
                if data.get("result") == "OK":
                    return True
            except (
                httpcore.ConnectError,
                json.decoder.JSONDecodeError,
                requests.exceptions.ConnectionError,
                ConnectionRefusedError,
                MaxRetryError,
                NewConnectionError,
                TimeoutError,
                OSError,
                LMStudioWebsocketError
                ):
                        nfo("|Ignorable| Source unavailable:", f"{api_name}")

    elif api_name == "cortex":
        try:
            response = requests.get("http://127.0.0.1:39281/v1/chat/completions", timeout=(1, 1))
            data     = response.json() if response is not None else {}
            if data.get("result") == "OK":
                return True
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
            nfo("|Ignorable| Source unavailable:", f"{api_name}")
    elif api_name == "llamafile":
        try:
            import openai
        except (ModuleNotFoundError, ImportError) as error_log:
            dbug(error_log)
        else:
            try:
                response = requests.get("http://localhost:8080/v1", timeout=(1, 1))
                data     = response.json() if response is not None else {}
                if data.get("result") == "OK":
                    return True
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
                nfo("|Ignorable| Source unavailable:", f"{api_name}")
    return False


class LibType(Enum):
    """API library constants"""

    OLLAMA   : int = (0, has_api("ollama"))
    HUB      : int = (1, has_api("huggingface_hub"))
    LM_STUDIO: int = (2, has_api("lmstudio"))
    CORTEX   : int = (3, has_api("cortex"))
    LLAMAFILE: int = (4, has_api("llamafile"))
    VLLM     : int = (5, has_api("vllm"))


class GenTypeC(BaseModel):
    """
    Generative inference types in ***C***-dimensional order\n
    ***Compute***, sorted from 'most involved' to 'least involved'\n
    :param clone: Copying identity, voice, exact mirror
    :param sync: Tone, tempo, color, quality, genre, scale, mood
    :param translate: A range of comprehensible approximations
    """

    clone    : Annotated[Callable | None, Field(default=None)]
    sync     : Annotated[Callable | None, Field(default=None)]
    translate: Annotated[Callable | None, Field(default=None)]


example_str = ("function_name", "import.function_name")


class GenTypeCText(BaseModel):
    """
    Generative inference types in ***C***-dimensional order for text operations\n
    ***Compute***, sorted from 'most involved' to 'least involved'\n
    :param research: Quoting, paraphrasing, and deriving from sources
    :param chain_of_thought: A performance of processing step-by-step
    :param question_answer: Basic, straightforward responses
    """

    research        : Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    chain_of_thought: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    question_answer : Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]


class GenTypeE(BaseModel):
    """
    Generative inference operation types in ***E***-dimensional order \n
    ***Equivalence***, lists sorted from 'highly-similar' to 'loosely correlated.'"\n
    :param universal: Affecting all conversions
    :param text: Text-only conversions
    """

    universal: GenTypeC     = GenTypeC(clone=None, sync=None, translate=None)
    text     : GenTypeCText = GenTypeCText(research=None, chain_of_thought=None, question_answer=None)


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
        ("text", "text") : ["llm", ("tool", True)],
    },
    LibType.HUB: {
        ("image", "text") : ["image-generation", "image-text-to-text", "visual-question-answering"],
        ("text", "text")  : ["chat", "conversational", "text-generation", "text2text-generation"],
        ("text", "video") : ["video generation"],
        ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
    },
}
