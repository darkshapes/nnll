# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

# courtesy HamenderSingh https://github.com/huggingface/diffusers/issues/2136#issuecomment-1952590982

from nnll.monitor.file import debug_monitor


@debug_monitor
def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """ "
    Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """

    import torch

    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
    negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)

    # create the tensor based on which prompt is longer
    if input_ids.shape[-1] >= negative_ids.shape[-1]:
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    else:
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length", max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i : i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i : i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
