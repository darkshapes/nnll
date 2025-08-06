# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

import sys
from typing import Any, Callable, Dict, List, Optional

from nnll.metadata.helpers import make_callable
from nnll.mir.doc_parser import parse_docs
from nnll.mir.json_cache import TEMPLATE_PATH_NAMED, JSONCache  # pylint:disable=no-name-in-module
from nnll.mir.mappers import cut_docs
from nnll.mir.tag import make_mir_tag
from nnll.tensor_pipe.deconstructors import root_class
from nnll.monitor.file import dbug

if "pytest" in sys.modules:
    import diffusers  # noqa # pyright:ignore[reportMissingImports] # pylint:disable=unused-import

nfo = print

TEMPLATE_FILE = JSONCache(TEMPLATE_PATH_NAMED)


@TEMPLATE_FILE.decorator
def flag_config(transformers: bool = False, data: dict = None, **kwargs):
    """Set type of MIR prefix depending on model type\n
    :param transformers: Use transformers data instead of diffusers data, defaults to False
    :raises ValueError: Model type not detected
    :return: _description_"""

    if transformers:
        flags = data["arch"]["transformer"]  # pylint:disable=unsubscriptable-object
    else:
        flags = data["arch"]["diffuser"]  # pylint:disable=unsubscriptable-object
    for mir_prefix, key_match in flags.items():
        if any(kwargs.get(param) for param in key_match):
            return mir_prefix
    return None
    # nfo(f"Unrecognized model type with {kwargs}\n" )


def diffusers_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Generate diffusion model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields
    """
    special_cases = {
        "black-forest-labs/FLUX.1-schnell": "black-forest-labs/FLUX.1-dev",
        "stabilityai/stable-diffusion-3-medium": "stabilityai/stable-diffusion-3.5-medium",
    }
    special_classes = {
        "StableDiffusion3Pipeline": "stabilityai/stable-diffusion-3.5-medium",  # NOT sd3
        "HunyuanDiTPipeline": "tencent-hunyuan/hunyuandiT-v1.2-diffusers",  #  NOT hyd .ckpt
        "ChromaPipeline": "lodestones/Chroma",
    }

    extracted_docs = list(cut_docs())
    pipe_data = {}  # pipeline_stable_diffusion_xl_inpaint
    for code_name, file_name, docs in extracted_docs:
        parse_result = parse_docs(docs)

        if parse_result:
            pipe_class = parse_result.pipe_class
            pipe_repo = parse_result.pipe_repo
            staged_class = parse_result.staged_class
            staged_repo = parse_result.staged_repo
            for class_name, swap_repo in special_classes.items():
                if pipe_class == class_name:
                    pipe_repo = swap_repo
                    break
            model_class_obj = make_callable(pipe_class, f"diffusers.pipelines.{code_name}.{file_name}")
            root_class(model_class_obj)
            try:
                series, comp_data = create_pipe_entry(pipe_repo, pipe_class)
            except TypeError:
                pass  # Attempt 1
            if pipe_data.get(series):
                exclude_list = ["Img2Img", "Control", "Controlnet"]  # causes issues with main repo resolution
                if any(maybe for maybe in exclude_list if maybe.lower() in pipe_class.lower()):
                    continue
            pipe_data.setdefault(series, {}).update(comp_data)
            if staged_class or pipe_repo in special_cases:
                test = special_cases.get(pipe_repo)
                if test:
                    staged_repo = test
                    staged_class = pipe_class
                try:
                    series, comp_data = create_pipe_entry(staged_repo or pipe_repo, staged_class or pipe_class)
                except TypeError:
                    continue  # Attempt 2,
                pipe_data.setdefault(series, {}).update(comp_data)
    return dict(pipe_data)


def create_pipe_entry(repo_path: str, class_name: str, model_class_obj: Optional[Callable] = None) -> tuple[str, Dict[str, Dict[Any, Any]]]:
    """Create a pipeline article and generate corresponding information according to the provided repo path and pipeline category\n
    :param repo_path (str): Repository path.
    :param model_class_obj (str): The model class function
    :raises TypeError: If 'repo_path' or 'class_name' are not set.
    :return: Tuple: The data structure containing mir_series and mir_comp is used for subsequent processing.
    """
    import diffusers  # pyright: ignore[reportMissingImports] # pylint:disable=redefined-outer-name

    mir_prefix = "info"
    if hasattr(diffusers, class_name):
        model_class_obj = getattr(diffusers, class_name)
        sub_segments = root_class(model_class_obj, "diffusers")
        decoder = "decoder" in sub_segments
        if repo_path in ["openai/shap-e", "kandinsky-community/kandinsky-3"]:
            mir_prefix = "info.unet"
        elif class_name == "MotionAdapter":
            mir_prefix = "info.lora"
        elif class_name == "WanPipeline":
            mir_prefix = "info.dit"
        else:
            mir_prefix = flag_config(**sub_segments)
            if mir_prefix is None and class_name not in ["AutoPipelineForImage2Image", "DiffusionPipeline"]:
                nfo(f"Failed to detect type for {class_name} {list(sub_segments)}\n")
            else:
                mir_prefix = "info." + mir_prefix
        mir_series, mir_comp = list(make_mir_tag(repo_path, decoder))
        mir_series = mir_prefix + "." + mir_series
        prefixed_data = {
            "repo": repo_path,
            "pkg": {0: {"diffusers": class_name}},
        }
        if class_name == "FluxPipeline":
            class_name = {1: {"mflux.flux.flux": "Flux1"}}
            prefixed_data["pkg"].update(class_name)
        elif class_name == "ChromaPipeline":
            class_name = {1: {"chroma": "ChromaPipeline"}}
            prefixed_data["pkg"].update(class_name)
        return mir_series, {mir_comp: prefixed_data}


def transformers_index():
    """Generate LLM model data for MIR index\n
    :return: Dictionary ready to be applied to MIR data fields"""

    import re

    import transformers

    from nnll.mir.mappers import stock_llm_data
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

    corrections: dict[dict[str, str | dict[str, list[str]]]] = {  # models with incorrect repos or config
        "BarkModel": {
            "repo_path": "suno/bark",
            "sub_segments": {"n_head": [""]},
        },
        "GraniteSpeechForConditionalGeneration": {
            "repo_path": "ibm-granite/granite-speech-3.3-8b",
            "sub_segments": {"encoder_layers": [""], "decoder_layers": [""]},
        },
        "GptOssModel": {
            "repo_path": "openai/gpt-oss",
        },
        "GraniteModel": {
            "repo_path": "ibm-granite/granite-3.3-2b-base",
            "sub_segments": {"rope_theta": [""]},
        },
        "DPRQuestionEncoder": {
            "repo_path": "facebook/dpr-question_encoder-single-nq-base",
            "sub_segments": {"local_attention": [""], "classifier_proj_size": [""]},
        },
        "CohereModel": {
            "repo_path": "CohereForAI/c4ai-command-r-v01",
            "sub_segments": {"attn_config": [""], "num_codebooks": [""]},
        },
        "Cohere2Model": {
            "repo_path": "CohereLabs/c4ai-command-r7b-12-2024",
            "sub_segments": {"attn_config": [""], "num_codebooks": [""]},
        },
        "GraniteMoeHybridModel": {
            "repo_path": "ibm-research/PowerMoE-3b",
        },
        "BertForMaskedLM": {
            "repo_path": "google-bert/bert-base-uncased",
        },
        "DistilBertModel": {
            "repo_path": "distilbert-base-uncased",
        },
        "GraniteMoeModel": {
            "repo_path": "ibm-research/PowerMoE-3b",
        },
        "AriaModel": {
            "repo_path": "rhymes-ai/Aria-Chat",
            "sub_segments": {"vision_config": [""], "text_config": [""]},
        },
        "TimmWrapperModel": {
            "repo_path": "timm/resnet18.a1_in1k",
            "sub_segments": {
                "_resnet_": [""],
            },
        },
    }

    mir_data = {}
    # transformers_data = stock_llm_data()
    transformers_data: Dict[Callable, List[str]] = stock_llm_data()
    for model_class_obj, model_data in transformers_data.items():
        class_name = model_class_obj.__name__
        if class_name in list(corrections):  # conditional correction from mappings above: `root_class` doesn't return anything in these cases
            repo_path = corrections[class_name]["repo_path"]
            sub_segments = corrections[class_name].get("sub_segments", root_class(model_data["config"][-1], "transformers"))
        else:
            repo_path = ""
            if model_data.get("config"):
                doc_attempt = [getattr(transformers, model_data["config"][-1]), model_class_obj.forward]
            for pattern in doc_attempt:
                doc_string = pattern.__doc__
                matches = re.findall(r"\[([^\]]+)\]", doc_string)
                if matches:
                    dbug(matches)
                    try:
                        repo_path = next(iter(snip.strip('"').strip() for snip in matches if "/" in snip))
                    except StopIteration as error_log:
                        nfo(f"ERROR >>{matches} : LOG >> {error_log}")
                        pass
                    break
            sub_segments: Dict[str, List[str]] = root_class(model_data["config"][-1], "transformers")
        if sub_segments and list(sub_segments) != ["kwargs"] and list(sub_segments) != ["use_cache", "kwargs"] and repo_path is not None:
            mir_prefix = flag_config(transformers=True, **sub_segments)
            if mir_prefix is None:
                nfo(f"Failed to detect type for {class_name} {list(sub_segments)}\n")
                continue
            else:
                mir_prefix = "info." + mir_prefix
            mir_suffix, mir_comp = list(make_mir_tag(repo_path))
            mir_series = mir_prefix + "." + mir_suffix
            tokenizer_class = tokenizer_class_from_name(class_name)
            mir_data.get("info.encoder.tokenizer", mir_data.setdefault("info.encoder.tokenizer", {})).update(
                {
                    mir_suffix: {
                        "pkg": {
                            0: {
                                "transformers": f"{tokenizer_class.__module__}.{tokenizer_class.__name__}",
                            },
                        }
                    }
                },
            )
            mir_data.setdefault(
                mir_series,
                {
                    mir_comp: {
                        "repo": repo_path,
                        "pkg": {0: {"transformers": class_name}},
                    },
                },
            )
    return mir_data


def mlx_repo_capture(base_repo: str = "mlx-community"):
    import os
    import re
    import mlx_audio

    result = {}
    result_2 = {}
    folder_path_named: str = os.path.dirname(mlx_audio.__file__)
    for root, dir, file_names in os.walk(folder_path_named):
        for file in file_names:
            if file.endswith((".py", ".html", ".md", ".ts")):
                with open(os.path.join(root, file), "r") as open_file:
                    content = open_file.read()
                    if "mlx-community/" in content:
                        matches = re.findall(base_repo + r'/(.*?)"', content)
                        for match in matches:
                            result[match] = f"{base_repo}/{match}"
                            previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
                            class_match = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
                            if class_match:
                                result_2[match] = {f"{base_repo}/{match}": [*class_match]}
                            else:
                                if os.path.basename(root) in ["tts", "sts"]:
                                    folder_name = match.partition("-")[0]
                                    file_path = os.path.join(root, "models", folder_name, folder_name + ".py")
                                    if os.path.exists(file_path):
                                        with open(file_path, "r") as model_file:
                                            read_data = model_file.read()
                                            class_match = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)

    return result_2


# def mlx_repo_capture(base_repo: str = "mlx-community"):
#     import os
#     import re
#     import mlx_audio

#     result = {}
#     result_2 = {}
#     folder_path_named: str = os.path.dirname(mlx_audio.__file__)
#     for root, _, file_names in os.walk(folder_path_named):
#         for file in file_names:
#             if file.endswith((".py", ".html", ".md", ".ts")):
#                 with open(os.path.join(root, file), "r") as open_file:
#                     content = open_file.read()
#                     if "mlx-community/" in content:
#                         matches = re.findall(base_repo + r'/(.*?)"', content)
#                         for match in matches:
#                             print(file)
#                             result[match] = f"{base_repo}/{match}"
#                             previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
#                             matches = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
#                             if matches:
#                                 result_2[match] = {f"{base_repo}/{match}": [*matches]}
#                             else:
#                                 result_2[match] = {f"{base_repo}/{match}": None}
#     return result_2


# def mlx_audio_scrape(base_repo: str = "mlx-community"):
#     import os
#     import re
#     import mlx_audio

#     result = {}
#     result_2 = {}
#     folder_path_named: str = os.path.dirname(mlx_audio.__file__)
#     for root, _, file_names in os.walk(folder_path_named):
#         for file in file_names:
#             if file.endswith((".py",)):
#                 with open(os.path.join(root, file), "r") as open_file:
#                     content = open_file.read()
#                     if "mlx-community/" in content:
#                         matches = re.findall(base_repo + r'/(.*?)"', content)
#                         for match in matches:
#                             result[match] = f"{base_repo}/{match}"
#                             previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
#                             matches = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
#                             if len(matches) > 1:
#                                 result_2[match] = {f"{base_repo}/{match}": [*matches]}
#                             else:
#                                 if "nn.Module" in content:
#                                     previous_data = content[content.rindex("nn.Module") - 50 : content.rindex("nn.Module")]
#                                     matches = re.search(r"(\w+)\.", previous_data, re.MULTILINE)
#                                     result_2[match] = {f"{base_repo}/{match}": [*matches]}
#     return result_2
