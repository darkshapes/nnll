from nnll_30 import read_json_file


def tk_count(model, message):
    # import tiktoken
    # encoding = tiktoken.get_encoding("cl100k_base")
    # token_count = len(encoding.encode(message))

    from litellm.utils import token_counter
    import os

    model_name = os.path.split(model)
    model_name = os.path.join(os.path.split(model_name[0])[-1], model_name[-1])
    return token_counter(model_name, text=message)
    # from ollama import embed

    # response = embed(model, input=message)
    # return len(response["embeddings"])


#
