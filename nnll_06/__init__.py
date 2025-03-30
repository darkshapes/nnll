#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


def litellm_counter(model, message):
    """
    Return token count of message based on model\n
    :param model: Model path to lookup tokenizer for
    :param message: Message to tokenize
    :return: `int` Number of tokens needed to represent message
    """
    from litellm.utils import token_counter
    import os

    model_name = os.path.split(model)
    model_name = os.path.join(os.path.split(model_name[0])[-1], model_name[-1])
    return token_counter(model_name, text=message)


def ollama_counter(model, message):
    """
    Return token count of message based on ollama model\n
    :param model: Model to lookup tokenizer for
    :param message: Message to tokenize
    :return: `int` Number of tokens needed to represent message
    """
    from ollama import embed

    response = embed(model, input=message)
    return len(response["embeddings"])


def tiktoken_counter(message, model="cl100k_base"):
    """
    Return token count of gpt based on model\n
    :param model: Model path to lookup tokenizer for
    :param message: Message to tokenize
    :return: `int` Number of tokens needed to represent message
    """
    import tiktoken

    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(message))
