import ollama


def ollama_models() -> dict:
    available_models = {}
    response: ollama.ListResponse = ollama.list()
    for model in response.models:
        available_models.setdefault(f"{model.model}-{(model.size.real / 1024 / 1024):.2f} MB", model.model)
    return available_models

    # if model.details:
    #     print("  Format:", model.details.format)
    #     print("  Family:", model.details.family)
    #     print("  Parameter Size:", model.details.parameter_size)
    #     print("  Quantization Level:", model.details.quantization_level)
    # print("\n")
