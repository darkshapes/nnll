#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import dspy

from nnll_01 import debug_monitor, debug_message as dbug
from nnll_15.constants import LibType


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 60 and 90 words and maximum 10000 character code blocks")


class ChainOfThought(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 30 and 50 words")
    # ChainOfThought("question -> answer")


class BasicHistory(dspy.Signature):
    message = dspy.InputField()
    history = dspy.InputField()
    answer = dspy.OutputField(desc="often between 60 and 90 words and maximum 10000 character code blocks")


class ChatMachine(dspy.Module):
    """Store previous responses as a list"""

    @debug_monitor
    async def __init__(self, memory_size: int = 5, signature: dspy.Signature = BasicHistory):
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
        self.generator = dspy.asyncify(dspy.Predict(signature))
        self.chat_instance = dspy.streamify(self.generator)

    @debug_monitor
    async def forward(self, message: str):
        self.memory.append(message)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        combined_context = " ".join(self.memory)
        async for chunk in self.chat_instance(message=message, history=combined_context, stream=True):
            try:
                if isinstance(chunk, dspy.Prediction):
                    if str(chunk) is not None:
                        yield chunk  # str(chunk)
                else:
                    chnk = chunk["choices"][0]["delta"]["content"]
                    if chnk is not None:
                        yield chnk
            except AttributeError as error_log:
                dbug(error_log)


@debug_monitor
async def chat_machine(library: LibType, model: str, message: str):
    kwargs = {}
    if library == LibType.OLLAMA:
        kwargs = {"model": f"ollama_chat/{model}", "api_base": "http://localhost:11434/api/chat", "model_type": "chat"}
    elif library == LibType.LM_STUDIO:
        kwargs = {"model": model, "api_base": "http://localhost:1234/v1", "api_key": "lm-studio"}
    elif library == LibType.HUB:
        kwargs = {}

    model = dspy.LM(**kwargs)
    dspy.settings.configure(lm=model, async_max_workers=8)
    machine = ChatMachine()
    yield machine.forward(message)
