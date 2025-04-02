from enum import Enum
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

mir_db = JSONCache(CONFIG_PATH_NAMED)


class LibType(Enum):
    """API library constants"""

    OLLAMA: int = 0
    HUB: int = 1
    VLLM: int = 2
    LM_STUDIO: int = 3


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
