### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from typing import Annotated, Callable, Optional

from pydantic import BaseModel, Field

from nnll_01 import dbug, nfo
from nnll_60 import JSONCache, LIBTYPE_PATH_NAMED

LIBTYPE_CONFIG = JSONCache(LIBTYPE_PATH_NAMED)


@LIBTYPE_CONFIG.decorator
def has_api(api_name: str, data: dict = None) -> bool:
    """Check available modules, try to import dynamically.
    True for successful import, else False

    :param api_name: Constant name for API
    :param _data: filled by config decorator, ignore, defaults to None
    :return: _description_
    """

    def check_host() -> bool:
        from json.decoder import JSONDecodeError
        import httpcore
        import httpx
        from urllib3.exceptions import NewConnectionError, MaxRetryError
        import requests

        dbug(f"Response from API  {api_data}")
        try:
            if api_data.get("api_url", 0):
                request = requests.get(api_data.get("api_url"), timeout=(1, 1))
                status = request.json() if request is not None else {}
                if status.get("result") == "OK":
                    return True
        except JSONDecodeError:
            dbug(f"json for ! {api_data}")
            dbug(request.status_code)
            if request.ok:
                return True
            else:
                try:
                    request.raise_for_status()
                except requests.HTTPError() as error_log:
                    dbug(error_log)

        except (requests.exceptions.ConnectionError, httpcore.ConnectError, httpx.ConnectError, ConnectionRefusedError, MaxRetryError, NewConnectionError, TimeoutError, OSError, RuntimeError, ConnectionError):
            nfo("|Ignorable| Source unavailable:", f"{api_name}")
            return False
        return False

    api_data = data[api_name]  # pylint: disable=unsubscriptable-object

    try:
        if api_name == "LM_STUDIO":
            from lmstudio import APIConnectionError, APITimeoutError, APIStatusError, LMStudioWebsocketError

            return check_host()
        elif api_name in ["LLAMAFILE", "CORTEX"]:
            from openai import APIConnectionError, APIStatusError, APITimeoutError

            return check_host()
        else:
            __import__(api_data.get("module"))
            if api_name == "HUB":
                return True
            else:
                return check_host()
    except (ImportError, ModuleNotFoundError):
        nfo("|Ignorable| Source unavailable:", f"{api_name}")
        return False
    except (APIConnectionError, APITimeoutError, APIStatusError):
        nfo("|Ignorable| Source unavailable:", f"{api_name}")
        return False
    except LMStudioWebsocketError:
        nfo("|Ignorable| Source unavailable:", f"{api_name}")
        return False
    return False


class LibType(Enum):
    """API library constants"""

    # Integers are usedto differentiate boolean condition
    # GIVEN : The state of all library modules & servers are marked at launch

    OLLAMA: tuple = (has_api("OLLAMA"), "OLLAMA")
    HUB: tuple = (has_api("HUB"), "HUB")
    LM_STUDIO: tuple = (has_api("LM_STUDIO"), "LM_STUDIO")
    CORTEX: tuple = (has_api("CORTEX"), "CORTEX")
    LLAMAFILE: tuple = (has_api("LLAMAFILE"), "LLAMAFILE")
    VLLM: tuple = (has_api("VLLM"), "VLLM")


example_str = ("function_name", "import.function_name")


class GenTypeC(BaseModel):
    """
    Generative inference types in ***C***-dimensional order\n
    ***Comprehensiveness***, sorted from 'most involved' to 'least involved'\n
    The terms define 'artistic' and ambiguous operations\n

    :param clone: Copying identity, voice, exact mirror
    :param sync: Tone, tempo, color, quality, genre, scale, mood
    :param translate: A range of comprehensible approximations\n
    """

    clone: Annotated[Callable | None, Field(default=None)]
    sync: Annotated[Callable | None, Field(default=None)]
    translate: Annotated[Callable | None, Field(default=None)]


class GenTypeCText(BaseModel):
    """
    Generative inference types in ***C***-dimensional order for text operations\n
    ***Comprehensiveness***, sorted from 'most involved' to 'least involved'\n
    The terms define 'concrete' and more rigid operations\n

    :param research: Quoting, paraphrasing, and deriving from sources
    :param chain_of_thought: A performance of processing step-by-step (similar to `reasoning`)
    :param question_answer: Basic, straightforward responses\n
    """

    research: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    chain_of_thought: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]
    question_answer: Annotated[Optional[Callable | None], Field(default=None, examples=example_str)]


class GenTypeE(BaseModel):
    """
    Generative inference operation types in ***E***-dimensional order \n
    ***Equivalence***, lists sorted from 'highly-similar' to 'loosely correlated.'"\n
    :param universal: Affecting all conversions
    :param text: Text-only conversions\n

    ***multimedia generation***
    ```
    Y-axis: Detail (Most involved to least involved)
    │
    │                             clone
    │                 sync
    │ translate
    │
    +───────────────────────────────────────> X-axis: Equivalence (Loosely correlated to highly similar)
    ```
    ***text generation***
    ```
    Y-axis: Detail (Most involved to least involved)
    │
    │                           research
    │             chain-of-thought
    │ question/answer
    │
    +───────────────────────────────────────> X-axis:  Equivalence (Loosely correlated to highly similar)
    ```

    This is essentially the translation operation of C types, and the mapping of them to E \n

    An abstract generalization of the set of all multimodal generative synthesis processes\n
    The sum of each coordinate pair reflects effective compute use\n
    In this way, both C types and their similarity are translatable, but not 1:1 identical\n
    Text is allowed to perform all 6 core operations. Other media perform only 3.\n
    """

    # note: `sync` may have better terms, such as 'harmonize' or 'attune'. `sync` was chosen because it is shorter

    universal: GenTypeC = GenTypeC(clone=None, sync=None, translate=None)
    text: GenTypeCText = GenTypeCText(research=None, chain_of_thought=None, question_answer=None)


# Here, a case could be made that tasks could be determined by filters, rather than graphing
# This is valid but, it offers no guarantees for difficult logic conditions that can be easily verified by graph algorithms
# Using graphs also allows us to offload the logic elsewhere

VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "3d_render", "vector_graphic", "upscale_image"]

# note : decide on a way to keep paired tuples and sets together inside config dict
VALID_TASKS = {
    LibType.CORTEX: {
        ("text", "text"): ["text"],
    },
    LibType.VLLM: {
        ("text", "text"): ["text"],
        ("image", "text"): ["vision"],
    },
    LibType.OLLAMA: {
        ("text", "text"): ["mllama", "llava", "vllm"],
    },
    LibType.LLAMAFILE: {
        ("text", "text"): ["text"],
    },
    LibType.LM_STUDIO: {
        # ("image", "text"): [("vision", True)],
        ("text", "text"): ["llm"],
    },
    LibType.HUB: {
        ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
        ("text", "text"): ["chat", "conversational", "text-generation", "text2text-generation"],
        ("text", "video"): ["video generation"],
        ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
    },
}
