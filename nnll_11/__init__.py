#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import dspy
from pydantic import BaseModel, Field

from nnll_01 import debug_monitor, debug_message as dbug
from nnll_15.constants import LibType


# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""

#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 60 and 90 words and maximum 10000 character code blocks")


# class ChainOfThought(dspy.Signature):
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 30 and 50 words")
#     # ChainOfThought("question -> answer")


class LLMInput(BaseModel):
    """Simple input query fields for chat models including previous contexts"""

    context: str = Field(description="The context for the question")
    query: str = Field(description="The question to be answered")


class LLMOutput(BaseModel):
    """Simple output fields for chat models
    Incl. confidence metric and 60-90 word/10k character code limits"""

    answer: str = Field(description="The answer for the question, often between 60 and 90 words and maximum 10000 character code blocks")
    confidence: float = Field(ge=0, le=1, description="The confidence score for the answer")


class BasicQAHistory(dspy.Signature):
    """Answer questions with short factoid answers."""

    message: LLMInput = dspy.InputField()
    response: LLMOutput = dspy.OutputField()


# class SigType(BaseModel):
#     QA: BasicQAHistory = BasicQAHistory


class ChatMachineWithMemory(dspy.Module):
    """Base module for Q/A chats using async and `dspy.Predict` List-based memory
    Defaults to 5 question history, 4 max workers, and `BasicQAHistory` query"""

    @debug_monitor
    def __init__(self, max_workers: int = 4, memory_size: int = 5, signature: dspy.Signature = BasicQAHistory):
        """
        Instantiate the module, setup parameters, create async streaming generator.\n
        Does not load any models until forward pass
        :param memory_size: The length of the memory
        :param signature: The format of messages sent to the model
        """
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
        self.max_workers = max_workers
        generator = dspy.asyncify(dspy.Predict(signature))
        self.completion = dspy.streamify(generator)

    @debug_monitor
    async def forward(self, api_config: dict, message: str):
        """
        Forward pass for LLM Chat process\n
        :param model: The library-specific arguments for the model configuration
        :param message: A simple string to send to the LLM
        :param max_workers: Maximum number of async processes
        :return: yields response in chunks
        """
        model = dspy.LM(**api_config)
        dspy.settings.configure(lm=model, async_max_workers=self.max_workers)
        self.memory.append(message)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        combined_context = " ".join(self.memory)
        async for chunk in self.completion(message={"context": combined_context, "query": message}, stream=True):
            yield chunk


async def start_machine(max_workers: int = 8, memory_size: int = 5, signature: dspy.Signature = BasicQAHistory):
    """
    Instantiate chat completion method based on library\n
    :param memory_size: Session chat history to retain for context
    :param signature: The type of dspy.Signature to use for completion
    :return: An instance of ChatMachineWithHistory
    """
    machine = ChatMachineWithMemory(memory_size=memory_size, max_workers=max_workers, signature=signature)
    return machine


@debug_monitor
async def chat_machine(library: LibType, machine: dspy.Module, message: str, model: str):
    """
    Load model into chat completion method based on library and run query\n
    :param model: The model to reply with
    :param message: The text to send the model
    :param library: API type to use to fulfill request
    :param machine: Module to run query through
    :return: Yields stream of `str` data in chunks
    """
    if library == LibType.OLLAMA:
        model = {"model": f"ollama_chat/{model}", "api_base": "http://localhost:11434/api/chat", "model_type": "chat"}
    elif library == LibType.LM_STUDIO:
        model = {"model": f"lm_studio/{model}", "api_base": "http://localhost:1234/v1", "api_key": "lm-studio"}
    elif library == LibType.HUB:
        model = {"model": f"huggingface/{model}"}  # , api_base="https://localhost:xxxx/address:port/sdbx/placeholder"}
    elif library == LibType.VLLM:
        model = {"model": f"hosted_vllm/{model}", "api_base": "http://localhost:8000/chat/completions"}
    async for chunk in machine.forward(api_config=model, message=message):
        try:
            if chunk is not None:
                if isinstance(chunk, dspy.Prediction):
                    yield chunk
                else:
                    chnk = chunk["choices"][0]["delta"]["content"]
                    if chnk is not None:
                        yield chnk
        except AttributeError as error_log:
            dbug(error_log)
