import asyncio

# import os
import dspy
# from pydantic import BaseModel


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


async def stream(message, generator):
    print(message)

    async def generate():
        async for chunk in generator(question=message):
            if isinstance(chunk, dspy.Prediction):
                response = chunk
                # print(response)
            else:
                response = chunk["choices"][0]["delta"]["content"]
                print(response)

    await generate()
    # print(f"Finished request from client {client_id}")


async def main(model, message):
    local_llama = dspy.LM(api_base="http://localhost:11434/api/chat", model=model, model_type="chat")
    dspy.settings.configure(lm=local_llama, async_max_workers=4)  # default is 8
    generator = dspy.asyncify(dspy.Predict(BasicQA))
    streaminator = dspy.streamify(generator)
    requests = [asyncio.create_task(stream(message, streaminator))]
    await asyncio.gather(*requests)


asyncio.run(
    main(
        model="ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M",
        message="So it was a coincidence that they became so similar?",
    )
)

# async def main():


# if __name__ == "__main__":
#     data = asyncio.run(main())
#     # print(f"{data}")

# Using await and/or return creates a coroutine function. To call a coroutine function, you must await it to get its results.
# await and return create coroutine functions. calls must await them
# yield in an async def block creates an asynchronous generator, which you iterate over with async for
# Anything defined with async def may not use yield from, which will raise a SyntaxError.

# async def f(x):
#     y = await z(x)  # OK - `await` and `return` allowed in coroutines
#     return y

# async def g(x):
#     yield x  # OK - this is an async generator

# async def m(x):
#     yield from gen(x)  # No - SyntaxError

# def m(x):
#     y = await z(x)  # Still no - SyntaxError (no `async def` here)
#     return y

# async def g():
#     # Pause here and come back to g() when f() is ready
#     r = await f()
#     return r

# # lm = dspy.LM("openai/gpt-4o-mini")
# class Question(BaseModel):
#     text: str


# async def chat(question: Question, program: dspy.Prediction):
#     async def generate():
#         async for chunk in program(question=question.text):
#             if isinstance(chunk, dspy.Prediction):
#                 response = {"prediction": chunk.labels().toDict()}
#             else:
#                 response = chunk["choices"][0]["delta"]["content"]
#             print(f"{response}")

#     return generate()


# async def main():
#     model = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"
#     local_llama = dspy.LM(api_base="http://localhost:11434/api/chat", model=model, model_type="chat")

#     dspy.settings.configure(lm=local_llama, async_max_workers=4)  # default is 8
#     dspy_program = dspy.asyncify(dspy.Prediction("question -> answer"))
#     program = dspy.streamify(dspy_program)
#     await asyncio.gather(chat("Testing code with a test question", program)) #any time consuming blocking call


# if __name__ == "__main__":
#     asyncio.run(main())


# import time


# def count():
#     print("One")
#     time.sleep(1)
#     print("Two")


# def main():
#     for _ in range(3):
#         count()


# if __name__ == "__main__":
#     s = time.perf_counter()
#     main()
#     elapsed = time.perf_counter() - s
#     print(f"{__file__} executed in {elapsed:0.2f} seconds.")


#!/usr/bin/env python3
# countsync.py


# async def stream(question: Question):
#     stream = streaming_dspy_program(question=question.text)
#     return StreamingResponse(streaming_response(stream), media_type="text/event-stream")


# def chat

# async def stream(question: message):
#     async def generate():
#         async for value in streaming_dspy_program(question=message.text):
#             if isinstance(value, dspy.Prediction):
#                 data = {"prediction": value.labels().toDict()}
#             elif isinstance(value, litellm.ModelResponse):
#                 data = {"chunk": value.json()}
#             yield f"data: {ujson.dumps(data)}\n\n"
#         yield "data: [DONE]\n\n"


# dspy.settings.configure(lm=local_llama)

# chat_module = dspy.streamify(dspy.Predict("message, history -> answer"))

# with dspy.context(lm=local_llama):
#     local_llama("Hello, just testing")
#     local_llama(messages=[{"role": "user", "content": "Test message"}])


# # async def use_streaming():
#     output_stream = program(input_text="write me a 1005 word essay")
#     # async for value in output_stream:
#     async for value in output_stream:
#       if isinstance(value, dspy.Prediction):
#         print(value)
#         print(value.output_text)
#       else:
#         print(value['choices'][0]['delta']['content'])  # Print each streamed value incrementally

# chat_module = ChatWithListMemory()

# metric = dspy.evaluate.SemanticF1(decompositional=True)


# async def use_streaming():
#     output_stream = program(input_text="write me a 1005 word essay")
#     # async for value in output_stream:
#     async for value in output_stream:
#         if isinstance(value, dspy.Prediction):
#             print(value)
#             print(value.output_text)
#         else:
#             print()  # Print each streamed value incrementally


# class ChatWithListMemory(dspy.Module):
#     def __init__(self, memory_size=5):
#         super().__init__()
#         self.memory = []
#         self.memory_size = memory_size
#         self.chat_instance = dspy.streamify(dspy.Predict("message, history -> answer"))

#     def forward(self, message: str):
#         self.memory.append(message)
#         if len(self.memory) > self.memory_size:
#             self.memory.pop(0)
#         combined_context = " ".join(self.memory)
#         response = self.chat_instance(message=message, history=combined_context)
#         return response


# async def chat(message):
#     response = chat_module(message)
#     output = None
#     async for value in response:
#         if isinstance(value, dspy.Prediction):
#             output = value["choices"][0]["delta"]["content"]
#             return output

# import dspy

# async def main():
#     client = ollama.AsyncClient()
#     for part in await client.generate(sys.args[0], sys.args[1], stream=True):
#         print(part['response'], end='', flush=True)

# async def send_request():
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print('\nGoodbye!')


# class ASession:
#     async def __aenter__(self):
#         print("Opening Session")
#         responses = [asyncio.create_task(tasks)]
#         await asyncio.gather(*responses)
#         return self

#     async def __aexit__(self, exc_type, exc, tb):
#         print("Closing Session...")
#         await asyncio.sleep(1)
