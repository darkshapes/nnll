import asyncio

# import os
import dspy
# from pydantic import BaseModel


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 30 and 50 words")


async def stream(message, streaminator):
    async for chunk in streaminator(question=message, stream=True):
        if isinstance(chunk, dspy.Prediction):
            yield chunk
        else:
            yield chunk["choices"]["delta"]["content"]


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


# async def main():
#     client = ollama.AsyncClient()
#     for part in await client.generate('llama3.2', 'Why is the sky blue?', stream=True):
#         print(part['response'], end='', flush=True)
