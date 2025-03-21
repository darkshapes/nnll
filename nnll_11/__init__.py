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


async def main(model, message):
    dspy.settings.configure(lm=model, async_max_workers=4)
    generator = dspy.asyncify(dspy.Predict(BasicQA))
    streaminator = dspy.streamify(generator)

    async for chunk in streaminator(question=message, stream=True):
        if isinstance(chunk, dspy.Prediction):
            if str(chunk) is not None:
                yield ""  # str(chunk)
        else:
            chnk = chunk["choices"][0]["delta"]["content"]
            if chnk is not None:
                yield chnk


async def chat_machine(model, message):
    async for chunk in main(model, message):
        yield chunk  # Stream each chunk in real-time


# ["choices"][0]["delta"]["content"]  # Process chunks as they arrive
# yield chunk if chunk is not none # Forward chunks for real-time streaming

# List all models available locally/ # List LLMs only # # List embeddings only
