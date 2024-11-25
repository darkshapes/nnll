
from nnll_11.src import method_crafter, encoder_classes, tokenizer_classes

def define_encoders(device, encoder_dict *paths):
    """
    ####  Iteratively create encoders and tokenizers
    #### `device`: processor to assign processing of model
    #### `encoder_dict`: the attributes to create `[*_class],[*_method],[*_location], [*_expressions]`
    #### `paths`: the locations of the model files
    #### OUTPUT: two `lists` of the models
    """
    for i, tk in enumerate(encoder_dict):
        setattr(tokenizer , f"_{i}" , method_crafter(tk.get("tokenizer_class"), tk.get("tokenizer_method"), tk.get("tokenizer_location"), tk.get("tokenizer_expressions")))
        setattr(text_encoder , f"_{i}" , method_crafter(tk.get("text_encoder_class"), tk.get("text_encoder_method"), tk.get("text_encoder_location"), tk.get("text_encoder_expressions")))

    return