from enum import Enum
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

mir_db = JSONCache(CONFIG_PATH_NAMED)


class LibType(Enum):
    ""

    OLLAMA: int = 0
    LM_STUDIO: int = 1
    HUB: int = 2


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
