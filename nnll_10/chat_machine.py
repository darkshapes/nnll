#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import dspy
# from pydantic import BaseModel


class ChatWithListMemory(dspy.Module):
    def __init__(self, memory_size=5):
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
        self.chat_instance = dspy.Predict("message, history -> answer")

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


async def stream(message, streaminator):
    async for chunk in streaminator(question=message, stream=True):
        if isinstance(chunk, dspy.Prediction):
            yield chunk
        else:
            yield chunk["choices"][0]["delta"]["content"]


async def main(model, message):
    local_llama = dspy.LM(api_base="http://localhost:11434/api/chat", model=model, model_type="chat")
    dspy.settings.configure(lm=local_llama, async_max_workers=4)
    generator = dspy.asyncify(dspy.Predict(BasicQA))
    streaminator = dspy.streamify(generator)

    async for chunk in stream(message, streaminator):  # Process chunks as they arrive
        yield chunk  # Forward chunks for real-time streaming


async def chat_machine(model, message):
    async for chunk in main(model, message):
        yield chunk  # Stream each chunk in real-time
