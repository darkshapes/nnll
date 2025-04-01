#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import dspy

from nnll_01 import debug_monitor, debug_message as dbug
from nnll_15.constants import LibType

# from pydantic import BaseModel


class ChatWithListMemory(dspy.Module):
    @debug_monitor
    def __init__(self, memory_size=5):
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
        self.chat_instance = dspy.Predict("message, history -> answer")

    @debug_monitor
    def forward(self, message: str):
        self.memory.append(message)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        combined_context = " ".join(self.memory)
        response = self.chat_instance(message=message, history=combined_context)
        return response


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 60 and 90 words and maximum 10000 character code blocks")


class ChainOfThought(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 30 and 50 words")
    # ChainOfThought("question -> answer")


@debug_monitor
async def chat_machine(model: str, message: str, library: LibType, **kwargs):
    if library == LibType.OLLAMA:
        kwargs = {
            "model": f"ollama_chat/{model}",
            "api_base": "http://localhost:11434/api/chat",
            "model_type": "chat",
        }
    elif library == LibType.LM_STUDIO:
        kwargs = {"model": model, "api_base": "http://localhost:1234/v1", "api_key": "lm-studio"}
    elif library == LibType.HUB:
        kwargs = {}

    model = dspy.LM(**kwargs)
    dspy.settings.configure(lm=model, async_max_workers=8)
    generator = dspy.asyncify(dspy.Predict(BasicQA))
    streaminator = dspy.streamify(generator)
    async for chunk in streaminator(question=message, stream=True):
        try:
            if chunk is not None:
                if not isinstance(chunk, dspy.Prediction):
                    chnk = chunk["choices"][0]["delta"]["content"]
                    yield chnk
                else:
                    dbug(chunk)
                    break
        except AttributeError as error_log:
            dbug(error_log)
