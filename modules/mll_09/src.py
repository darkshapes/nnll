
import os
import sys
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if modules_path not in sys.path:
    sys.path.append(modules_path)

from mll_11.src import method_crafter, encoder_classes, tokenizer_classes

def encode_prompt(prompts, tokenizers, text_encoders):
    """
    ####  Create prompt encodings prior to loading main model to lower memory overhead
    #### `prompts`: prompt string fed from a list of prompts
    #### `tokenizers`: tokenizer models
    #### OUTPUT: a set of encodings formatted to feed into `pipe`
    """
    embeddings_list = []
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        prompt_embeds = text_encoder(cond_input.input_ids.to("cuda"), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        embeddings_list.append(prompt_embeds.hidden_states[-2])

        prompt_embeds = torch.concat(embeddings_list, dim=-1)

    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def define_encoders(device, tokenizers, text_encoders, *paths):
    """
    ####  Iteratively create encoders and tokenizers
    #### `device`: processor to assign processing of model
    #### `tokenizers`: the tokenizer models to create
    #### `text_encoders`: the text_encoders to create
    #### `paths`: the locations of the model files
    #### OUTPUT: two `lists` of the models
    """
    for i, tk in enumerate(tokenizers):
        setattr(tokenizer,f"_{i}",(tk.get("key_class"), tk.get("method"), tk.get("name"), tk.get("args"))

    for i, te in enumerate(text_encoders):
        setattr(text_encoder,f"_{i}",method_crafter(te.get("key_class"), te.get("method"), te.get("name"), te.get("args")))

    tokenizer = CLIPTokenizer.from_pretrained(
        clip,
        local_files_only=True,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        clip,
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant='fp16',
        local_files_only=True,
    ).to(device)

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        clip2,
        local_files_only=True,
    )

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        clip2,
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant='fp16',
        local_files_only=True,
    ).to(device)
    return [tokenizer, tokenizer_2], [text_encoder, text_encoder_2]

def create_encodings(device, queue, paths):
    tokenizer_models, text_encoder_models = define_encoders(paths, device)
    with torch.no_grad():
        for generation in queue:
            generation['embeddings'] = encode_prompt(
                [generation['prompt'], generation['prompt']],
                tokenizer_models, text_encoder_models
            )
    return queue

    del tokenizer, text_encoder, tokenizer_2, text_encoder_2
