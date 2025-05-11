### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def encode_prompt(device, prompts, tokenizers, text_encoders):
    """
     Create prompt encodings prior to loading main model to lower memory overhead
    :param `prompts`: Prompt string fed from a list of prompts
    :param `tokenizers`: Tokenizer models
    OUTPUT: a set of encodings formatted to feed into `pipe`
    """
    import os
    import sys
    import torch
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    embeddings_list = []
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        prompt_embeds = text_encoder(cond_input.input_ids.to(device), output_hidden_states=True)

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
