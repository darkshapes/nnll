#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import dspy
from pydantic import BaseModel

from nnll_01 import debug_monitor

#         async def process text chat


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
    # Predict(BasicQA)


class ChainOfThought(dspy.Signature):
    """Answer questions showing work in progress"""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 30 and 50 words")
    # ChainOfThought("question -> answer")


class dspyConstructor(BaseModel):
    """Load model and create requests to known inference endpoints via dspy"""

    active_model: dspy.LM = None
    generator: dspy.asyncify = None
    streaminator: dspy.asyncify = None

    @debug_monitor
    async def link_model(self, model: str, api_base: str, **kwargs):
        """Set model for inference"""
        self.active_model = dspy.LM(api_base=api_base, model=model, **kwargs)
        dspy.settings.configure(lm=self.active_model, async_max_workers=4)

    @debug_monitor
    async def load_ollama(self, model: str):
        """Ollama specific implementation"""
        await self.link_model(model, "http://localhost:11434/api/chat")

    @debug_monitor
    async def load_lmstudio(self, model: str):
        """Lmstudio specific implementation"""
        await self.link_model(model, "http://localhost:1234/v1", api_key="lm-studio")

    @debug_monitor
    async def initialize_generator(self, component: dspy.Signature):
        """Add generator with inference type"""
        self.generator = dspy.asyncify(dspy.Predict(component))
        self.streaminator = dspy.streamify(self.generator)

    @debug_monitor
    async def respond_to(self, message: str):
        """Create a chat response from user message"""
        async for chunk in self.streaminator(question=message, stream=True):
            if isinstance(chunk, dspy.Prediction):
                if str(chunk) is not None:
                    yield ""  # Returns all text AFTER generation (Prediction("x"))
            else:
                chnk = chunk["choices"][0]["delta"]["content"]
                if chnk is not None:
                    yield chnk  # Streams text
