#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import dspy
from pydantic import BaseModel, Field

from nnll_01 import debug_monitor, debug_message as dbug  # , info_message as nfo
from nnll_15.constants import LibType


# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""

#     question = dspy.InputField()
#     answer = dspy.OutputField()


# class ChainOfThought(dspy.Signature):
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 30 and 50 words")
#     # ChainOfThought("question -> answer")


class LLMInput(BaseModel):
    """Simple input query fields for chat models including previous contexts"""

    context: str = Field(description="The context for the question")
    query: str = Field(description="The message to respond to")


# class LLMOutput(BaseModel):
#     """Simple output fields for chat models
#     Incl. confidence metric"""

#     reply: str = Field(description="The response to the question")
#     # confidence: float = Field(ge=0.0, le=1.0, description="The confidence score for the reply (absolute certainty is impossible).")  # Alternatively : ""Mean numeric value of conflicting predicates and cognitive dissonance for prediction per word."


class BasicQAHistory(dspy.Signature):
    """Reply with short responses within 60-90 word/10k character code limits"""

    message: LLMInput = dspy.InputField()
    response = dspy.OutputField(desc="Often between 60 and 90 words and limited to 10000 character code blocks")


@debug_monitor
async def get_api(model: str, library: LibType) -> dict:
    """
    Load model into chat completion method based on library and run query\n
    :param model: The model to create a reply with the question
    :param library: API Library to use
    :return: Arguments to pass to the LM constructor
    """

    if library == LibType.OLLAMA:
        model = {"model": model, "api_base": "http://localhost:11434/api/chat", "model_type": "chat"}  # ollama_chat/
    elif library == LibType.LM_STUDIO:
        model = {"model": model, "api_base": "http://localhost:1234/v1", "api_key": "lm-studio"}  # lm_studio/
    elif library == LibType.HUB:
        model = {"model": model}  # api_base="https://localhost:xxxx/address:port/sdbx/placeholder"} # huggingface/
    elif library == LibType.VLLM:
        model = {"model": model, "api_base": "http://localhost:8000/chat/completions"}  # hosted_vllm/
    return model


class ChatMachineWithMemory(dspy.Module):
    """Base module for Q/A chats using async and `dspy.Predict` List-based memory
    Defaults to 5 question history, 4 max workers, and `BasicQAHistory` query"""

    @debug_monitor
    def __init__(self, memory_size: int = 5, signature: dspy.Signature = BasicQAHistory):
        """
        Instantiate the module, setup parameters, create async streaming generator.\n
        Does not load any models until forward pass
        :param memory_size: The length of the memory
        :param signature: The format of messages sent to the model
        """
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
        generator = dspy.asyncify(dspy.Predict(signature))
        self.completion = dspy.streamify(generator)

    @debug_monitor
    async def forward(self, message: str, model: str, library: LibType, max_workers=4):
        """
        Forward pass for LLM Chat process\n
        :param model: The library-specific arguments for the model configuration
        :param message: A simple string to send to the LLM
        :param max_workers: Maximum number of async processes
        :return: yields response in chunks
        """
        from nnll_05 import lookup_function_for

        if library == LibType.HUB:
            constructor = await lookup_function_for(model)
            async for chunk in constructor(model):
                yield chunk
        else:
            api_kwargs = await get_api(model=model, library=library)
            model = dspy.LM(**api_kwargs)
            dspy.settings.configure(lm=model, async_max_workers=max_workers)
            combined_context = " ".join(self.memory)
            message = message["text"]
            self.memory.append(message)
            if len(self.memory) == self.memory_size:
                self.memory.pop(0)
            async for chunk in self.completion(message={"context": combined_context, "query": message}, stream=True):
                try:
                    if chunk is not None:
                        if isinstance(chunk, dspy.Prediction):
                            pass
                            # yield str(chunk)
                        else:
                            yield chunk["choices"][0]["delta"]["content"]
                except (GeneratorExit, RuntimeError, AttributeError) as error_log:
                    dbug(error_log)  # consider threading user selection between cursor jumps
