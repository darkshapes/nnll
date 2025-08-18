# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""自動化索引"""
# regex to remove \[[^\]]*\]
# (?m)^\s*"[^"]+"(?=\s*:)
# (?m)^\s*"[^"]+"\s?:
# modelspec sai https://github.com/Stability-AI/ModelSpec

from logging import INFO, Logger
from typing import Dict, List, Tuple

from nnll.mir.maid import MIRDatabase
from nnll.mir.mir import mir_entry
from nnll.mir.tag import make_mir_tag
from nnll.monitor.file import dbug, dbuq

nfo_obj = Logger(INFO)
nfo = nfo_obj.info

sd1_series, sd1_comp = make_mir_tag("stable-diffusion-v1-5/stable-diffusion-v1-5")
sdxl_series, sdxl_comp = make_mir_tag("stabilityai/stable-diffusion-xl-base-1.0")
dev_series, dev_comp = make_mir_tag("black-forest-labs/FLUX.1-dev")
schnell_series, schnell_comp = make_mir_tag("black-forest-labs/FLUX.1-schnell")
ssd_series, ssd_comp = make_mir_tag("segmind/SSD-1B")
vega_series, vega_comp = make_mir_tag("segmind/Segmind-Vega")
sd3_series, sd3_comp = make_mir_tag("stable-diffusion-3.5-medium")  #

# def gen_attention_processors(mir_db: MIRDatabase): # upstream not quite ready for this yet
#     from diffusers.models.attention_processor import AttentionProcessor

#     mir_data
#     for series, comp_name in mir_data.items():
#     id_segment = series.split(".")
#     for compatibility in comp_name:
#         dbug(id_segment)
#         try:
#             mir_db.add(
#                 mir_entry(
#                     domain=id_segment[0],
#                     arch=id_segment[1],
#                     series=id_segment[2],
#                     comp=compatibility,
#                     **mir_data[series][compatibility],
#                 ),
#             )
#         except IndexError as error_log:
#             nfo(f"Failed to create series: {series}  compatibility: {comp_name}  ")
#             dbug(error_log)


# def gen_guiders(mir_db: MIRDatabase):  # upstream not quite ready for this yet
#     from nnll.metadata.helpers import snake_caseify
#     from diffusers.guider import GuiderType

#     guider_type = GuiderType
#     for comp_name in guider_type.items():
#         class_obj = comp_name.__name__
#         mir_data = {"pkg": {0: {"diffusers": class_obj}}}
#         try:
#             mir_db.add(
#                 mir_entry(
#                     domain="ops",
#                     arch="noise_prediction",
#                     series="guider",
#                     comp=snake_caseify(class_obj),
#                     **mir_data,
#                 ),
#             )
#         except IndexError as error_log:
#             nfo(f"Failed to create compatibility: {class_obj}")
#             dbug(error_log)


# (
#     "info.unet",
#     "stable-cascade",
#     {
#         "combined": {
#             "pkg": {
#                 0: {  # decoder=decoder_unet
#                     "precision": "ops.precision.bfloat.B16",
#                     "generation": {
#                         "negative_prompt": "",
#                         "num_inference_steps": 20,
#                         "guidance_scale": 4.0,
#                         "num_images_per_prompt": 1,
#                         "width": 1024,
#                         "height": 1024,
#                     },
#                 },
#                 "pkg_alt": {
#                     0: {
#                         "diffusers": {
#                             "StableCascadeCombinedPipeline": {
#                                 "negative_prompt": "",
#                                 "num_inference_steps": 10,
#                                 "prior_num_inference_steps": 20,
#                                 "prior_guidance_scale": 3.0,
#                             }
#                         },
#                     }
#                 },
#             }
#         }
#     },
# ),


def assimilate(mir_db: MIRDatabase, data_tuple: List[Tuple[Dict[str, any]]]) -> None:
    """Merge new data into a pre-generated MIR database, updating while preserving existing data structures.\n
    :param mir_db: The MIRDatabase instance
    :param data_tuple: A list of tuples, each containing:\n
            - arch (str): The architecture name
            - series (str): The series name
            - `new_data`: New data to be merged into the database.
    :raises TypeError: If any field in `new_data` is not a dictionary.
    """

    def update_nested_dict(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                if isinstance(target, dict):
                    update_nested_dict(target[key], value)
            else:
                if isinstance(source, dict):
                    # dbuq(target)
                    target.setdefault(key, value)
                else:
                    target = {key: value}

    dbug(f"{data_tuple}, {len(data_tuple)}")
    for arch, series, new_data in data_tuple:
        mir_data = mir_db.database[f"{arch}.{series}"]
        for comp, field_data in new_data.items():
            if not isinstance(field_data, dict):
                raise TypeError(f"{field_data} <-- Cannot combine with database: Not `dict()`")

            # dbuq(f"{arch}.{series} : {comp}")
            update_nested_dict(mir_data.setdefault(comp, {}), field_data)

            if series == sdxl_series:
                for field, field_data in field_data.items():
                    if isinstance(field_data, dict):
                        for definition, sub_def_data in field_data.items():
                            # dbug(definition)
                            if isinstance(sub_def_data, dict):
                                mir_data[comp][field].setdefault(definition, {})
                                update_nested_dict(mir_data[comp][field][definition], sub_def_data)


def hf_pkg_to_mir(mir_db: MIRDatabase):
    """Generate MIR HF Hub model database"""

    from nnll.mir.indexers import diffusers_index, transformers_index

    mir_data = diffusers_index() | transformers_index()
    for series, comp_name in mir_data.items():
        id_segment = series.split(".")
        for compatibility in comp_name:
            # dbug(id_segment)
            try:
                mir_db.add(
                    mir_entry(
                        domain=id_segment[0],
                        arch=id_segment[1],
                        series=id_segment[2],
                        comp=compatibility,
                        **mir_data[series][compatibility],
                    ),
                )
            except IndexError:  # as error_log:
                nfo(f"Failed to create series: {series}  compatibility: {comp_name}  ")
                # dbug(error_log)


def add_mir_dtype(mir_db: MIRDatabase):
    """Create mir info database"""
    import re

    import torch

    from nnll.metadata.helpers import slice_number

    available_dtypes: List[str] = [dtype for dtype in torch.__dict__.values() if isinstance(dtype, torch.dtype)]
    series_name = "_"
    for precision in available_dtypes:
        dep_name, class_name = str(precision).split(".")
        if "_" in class_name:
            comp_name = class_name[0].upper() + "8_" + class_name.split("_")[1].upper()
            if comp_name.endswith("FN"):
                comp_name = comp_name[:-2]
        else:
            comp_name = class_name[0].upper() + str(slice_number(class_name))
        variant_name = class_name.replace("bfloat", "bf").replace("float", "fp")
        dbuq(variant_name)
        patterns = [r"complex", r"bits", r"quint", r"uint", r"int", r"bfloat", r"float", r"bool"]
        for precision_name in patterns:
            compiled = re.compile(precision_name)
            dtype = re.search(compiled, class_name)
            if dtype:
                series_name = dtype.group()
                break

        mir_db.add(
            mir_entry(
                domain="ops",
                arch="precision",
                series=series_name,
                comp=comp_name,
                pkg={0: {dep_name.lower(): {class_name.lower(): {"variant": variant_name}}}},
            )
        )


def add_mir_schedulers(mir_db: MIRDatabase):
    """Create mir info database"""
    from nnll.mir.tag import make_scheduler_tag

    # from nnll.tensor_pipe.parenting import seek_class_path
    from importlib import import_module

    from diffusers import _import_structure

    for class_name in _import_structure["schedulers"]:
        if class_name != "SchedulerMixin":
            series_name, comp_name = make_scheduler_tag(class_name)
            class_obj = import_module("diffusers.schedulers")
            class_path = getattr(class_obj, class_name).__module__
            class_path = class_obj.__module__
            mir_db.add(
                mir_entry(
                    domain="ops",
                    arch="scheduler",
                    series=series_name,
                    comp=comp_name.lower(),
                    pkg={0: {"diffusers": class_name}},
                )
            )

    class_name = "KarrasDiffusionSchedulers"
    series_name, comp_name = make_scheduler_tag(class_name)
    class_obj = import_module("diffusers.schedulers.scheduling_utils")
    class_path = getattr(class_obj, class_name).__module__
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series=series_name,
            comp=comp_name,
            pkg={
                0: {"diffusers": class_name},
            },
        ),
    )


# def auto_gan etc etc
# ai-forever/Real-ESRGAN


def mir_update(mir_db: MIRDatabase, task_list: list = None, pipe_list: list = None):
    """Create mir unet info database"""
    from nnll.mir.tag import tag_pipe, tag_base_model

    diffusers_addons = [
        (
            "stabilityai/stable-diffusion-xl-base-1.0",
            "StableDiffusionXLPipeline",
            {
                "pkg": {
                    0: {
                        "generation": {
                            "denoising_end": 0.8,
                            "num_inference_steps": 40,
                            "output_type": "latent",
                            "safety_checker": False,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                    1: {"diffusers": "DiffusionPipeline"},
                },
                "file_256": [
                    "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",  # modelspec sai
                    "e6bb9ea85bbf7bf6478a7c6d18b71246f22e95d41bcdd80ed40aa212c33cfeff",  # modelspec sai vae 0.9
                    "357650fbfb3c7b4d94c1f5fd7664da819ad1ff5a839430484b4ec422d03f710a",  # diffusers
                    "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139",  # fp16 diffusers
                ],
                "layer_256": [
                    "62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef",  # any modelspec sai
                    "34dff8d98898baa0f10e71943e56b588cc114253b0d2f1051f3ce7a8a45fee0b",  # diffusers
                    "56b1ccd89b0d6ab658048aa34d659788b6ed663f13ef566f4b11bccef590b9da",  # diffusers fp16
                ],
                "layer_b3": [
                    "8be44fa13c1efa60f8bcadaa57f1d718473f9660f03c4f0e65dc037960d8cba1",  # any modelspec sai
                    "c9ab95ed1851418b65ef99651c1eb6bbdd2e3b0715e0e435d6d1e56ce310fac3",  # diffusers
                    "adfa260098d87616d748e3cf9c10bb2c90ff8890a84abbb2853d4aa69664070b",  # diffusers fp16
                ],
                "identifiers": ["logit_scale", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight", "add_embedding.linear_2.bias"],
            },
        ),
        (
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            "StableDiffusionXLImg2ImgPipeline",
            {
                "pkg": {
                    1: {
                        "diffusers": "DiffusionPipeline",
                        "generation": {"num_inference_steps": 40, "denoising_end": 0.8},
                    }
                },
                "identifiers": ["conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"],
                "file_256": [
                    "54f9cd2f2daf3aeec0b2708fa3dbc0e84e4f8ddd1ddead42e5bc60c6572c989f",  # diffusers
                    "7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f",  # modelspec sai
                    "3ea0376dcf065eaefd27806394a90e310001b1a71d4f1cf1f655e86c0e566ffe",  # fp16 diffusers
                ],
                "layer_b3": [
                    "6281355dbb37e5769c9460ae0ac75506d89932e2f97b09d9ade32ecf191e75ba",
                    "afb0639aae2eb65577c12d4a30cf7c9b3620ae63ba64a8fa632b58608c8a7a2e",
                    "669046014b69d98ab0f6fbb59547644436e0275f8b638f467ce2a873c3313683",
                ],
                "layer_256": [
                    "bb9eadbfabb52c0d8645783525a3fa70b59e9d7d09d5290d742a303262e793a2",
                    "c5adb56fe51343af2c3d493eb9f41515c204bd91eb9f40b983d45f70a1fa3b6d",
                    "1f838e39ed6e916258aee6990b72c09b34aa8eb3b5342234a497b8852b3df1c6",
                ],
            },
        ),
        (
            "lodestones/Chroma",
            "ChromaPipeline",
            {
                "pkg": {
                    1: {
                        "generation": {"neg_text": "", "num_steps": "28", "latent_size": [64, 64]},
                    }
                },
                "file_256": [
                    "53adcb3b6b6005758d40e2d8058b044ed4892bc8616efb7a62cc2dd384be07de",  # v1
                    "2c41e8a9831f3be1eaff2c2ed590abb62e4534e814f7ec58a5fd74ff71dc2036",  # v46,
                    "0a7b2d9699dbd22b3744ee2692900cabcfb731a43dac13729c33807f2bb7c9f6",  # v37 detail
                    "6ddc9e2bbe3376ab5ee9f10b2d947f127b6bf6f879f06f316a2208bb0da357b8",  # mlx chroma / v36 detail
                ],
                "layer_b3": [
                    "15e227ced8a89c41abaa9cc44f84dfffdf5ead0c626035e5a2dde2bbb0935479",
                ],
                "layer_256": ["a4daa6ff6f45ca70c738adb8c19bc3b6f228df931e6bf2a3394463e4dd7ec882"],
            },
        ),
        (
            "fal/AuraFlow",
            "AuraFlowPipeline",
            {
                "identifiers": [[8192, 3072], "mlpX.c_fc2.weight", "joint_transformer_blocks.2.ff_context.linear_2.weight"],
                "file_256": [
                    "ce3e475246258b94ee9dcb8b83292cb34edfffc2bbde46c74604d9c6cd7c585c",
                    "526be97cf581c89ad87c6b19c1f7c2378851137698f7ec436596d061a382d37b",  # sai
                    "6a40b011f287452dbca80face78e667055904c5ad97eb2097ade3200259b2203",  # diffusers fp16
                    "05e5493018333d947bb5940083dbc2f071093027ff414bc5b1b1229e4836e5cb",  # diffusers
                ],
                "layer_b3": [
                    "cc6d383576c35a9709798d2e2b9e3eb31ba8c608040cf3712bc37871cfd14e21",
                    "ddd54c44fa28fbddecf7cfae91cfa04917fd2f2fa94fc78c528cef2356a4ec3a",  # sai
                    "90c694e7d1e20e6da49b571e9954338d384775419790be315304103227b1051b",
                    "9e85aec1bdb616f52f88c80ddc7ab1eae8c16c0b5fbfcdb61a71ac02c325003d",
                ],
                "layer_256": [
                    "3c13e6a965d03a49227d8b1606ba6a343a23772d8768407cc78d4ddb9102bc80",
                    "b356cc84a23bc93bda4cc0fce1d0ba1b8e3d5a521e659ffc72e9e4a2d2c7f204",
                    "270df7317fe01abf06333acbbd4f15f8fc7a7c56053219f42efb598454a3af24",
                    "7ab6aa4514dd09f3cf589587d51a81734193ce45dd51bda9db0bd62fe48ef7d5",
                ],
            },
        ),
        (
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
            "HunyuanDiTPipeline",
            {
                "identifiers": ["extra_embedder", "model.blocks", "skip_norm.weight"],
                "file_256": [
                    "4fb84f84079cda457d171b3c6b15d1be95b5a3e5d9825703951a99ddf92d1787",  # normal
                    "e01db5e129e8ca1117e9cf473fc5a2b096949f03ab90048aeabbc328de7ec800",  # distilled
                    "8af691cadb78047d55721259355d708e87ddbba1b7845df9377d9a5ae917b45d",  # 1.2
                ],
                "layer_b3": [
                    "aead6b61b17ebc77c4c186a4b82c193f11ec267b20d909726422ee9852e2e0b2",
                    "885a056b94f6f9844c0660be489844d63bb74cc13316f441d10968fff3dd3120",  # distilled
                    "390d951cbdda6e2cffb690031b60f02921624651534c2effaaa7d68ab476c700",
                ],
                "layer_256": [
                    "d4842ce2b7f927203326b25ff4d6738ec9a8b95327f06791c387e4a351ed6ed0",
                    "5af943f96f5dc9fecb1e92fe2b1fa17c94dd6947690201f4a5ee1a4a2721a68e",  # distilled
                    "4a1f2b8234fa4336e263842e042d42e8d64d8a4d3941d9c0c78366b50303950c",  # 1.2
                ],
            },
        ),
        (
            "Alpha-VLLM/Lumina-Next-SFT-diffusers",
            "LuminaPipeline",
            {
                "pkg": {
                    0: {
                        "precision": " ops.precision.bfloat.B16",
                    },
                },
                "identifiers": ["time_caption", "feed_forward"],
                "file_256": [
                    "371153b7c7b7a64899d4016970c7cc472039f9c9b21ebe073adf0b8525cdf1bd",
                ],
                "layer_b3": [
                    "fa134efd6e9672e7de2965e4895fc58879bd0a6c4fdf9165c278f2748254675f",
                    "4d960ec35c53f72f065b94b836bcd923ea6074d38ad49881061f315d62e3c839",
                ],
                "layer_256": [
                    "3938a85568d9df186923edf04391d79e89e6199123bc175afb520e0948d1ae05",
                    "c0ca51fdea051fcd042bf4b56d32e1e8bb9525a921f2e197f370f101e90527f0",
                ],
            },
        ),
        (
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "PixArtSigmaPipeline",
            {
                "identifiers": ["adaln_single", "scale_shift_table"],
                "file_256": [
                    "c34b520ef473329b945c2a21083cdf1337c5a468d23b3215b65576789bfd0305",
                    "2fa4dee9229c02b03163f57bdb8e80c7a5ee364b7161796abe9c05e8dd13f239",
                ],
                "layer_b3": [
                    "a199930ff537994872da77391955f0dd52eddd22ab9105388f0c5852f1b8021f",
                    "ee6f980c32e98da6885f3e97d3f88d9158031e362cd3a49b20d1e23924b251e3",
                ],
                "layer_256": [
                    "e0afd203aff5a1d192e325d0f59361373273d85d138b51768c3f10a75c154dc0",
                    "987f3c2ff5d399191e5fd7dd7b1f1f285c197dc8124ad77f05cde7f2fb677a3c",
                ],
            },
        ),
        (
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "PixArtAlphaPipeline",
            {
                "identifiers": ["aspect_ratio", "y_embedding", "emb.resolution", "caption_projection"],
                "file_256": ["809a92d52a4a228f381a4b4f4b76051294b73285fb0cbb02f0ad24f9372217a8"],
                "layer_b3": ["c5be83545ce9dbc564bcc9fd8fe4157d131347ccfc8f62adc877ec205b20acee"],
                "layer_256": ["117225c0e91423746114b23d3e409708ad55c90ff52b21fa7a1c5105d2e935a5"],
            },
        ),
        (
            "stabilityai/stable-diffusion-3.5-medium",
            "StableDiffusion3Pipeline",
            {
                "pkg": {
                    0: {"precision": "ops.precision.float.F16"},
                },
                "identifiers": ["model.diffusion_model.joint_blocks.", "transformer_blocks.21.norm1_context.linear.weight", "transformer_blocks.31.norm1_context.linear.weight", "blocks.11.ff.net.2.weight"],
                "file_256": [
                    "ffef7a279d9134626e6ce0d494fba84fc1c7e720b3c7df2d19a09dc3796d8f93",  # large
                    "11fe06e22364b823dfeedc275912336b932b32a293a0b2f35ffac071990cc4de",  # medium
                ],
                "layer_b3": [
                    "e411016545785046810b29cc3999f40bc6392be134a1318386c6f1c48f98726a",
                    "a81e07ee67bc627e8b3c5e292ec1ca239009517a2106e8249d670ced0a88f746",  # med
                ],
                "layer_256": [
                    "13c982a6dc82d21c9f459e837d8c6f6d4696fd6e7e7b5783bdd2250b1f4fec61",
                    "6ee79050373337bf63ac20916596df778bb22022bb38af986128a7459eda1463",  # med
                ],
            },
        ),
        (
            "Efficient-Large-Model/Sana-1600M-1024px-BF16-diffusers",
            "SanaPipeline",
            {
                "pkg": {
                    0: {
                        "generation": {
                            "height": 1024,
                            "width": 1024,
                            "guidance_scale": 4.5,
                            "num_inference_steps": 20,
                        },
                        "precision": "ops.precision.bfloat.B16",
                    },
                },
                "file_256": [
                    "b0b50c33be8758713459aa3c760feef6315d4bea31521fb5b8c3e8fdd9841ffe",
                ],
                "layer_b3": [
                    "461e3d83dfa7e075ef21e2138ef153922ecfadde3db464b03dff92819f3e86dd",
                ],
                "layer_256": [
                    "b928bbcc2ce99d55d21c189e2b1c57498bc313ef5b1457036e356107d567fc4e",
                ],
            },
        ),
        (
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "StableDiffusionPipeline",
            {
                "identifiers": ["up_blocks.3.attentions.0.transformer_blocks.0.norm3.weight"],
                "file_256": [
                    "6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa",  # pruned ema only original safetensors
                    "1a189f0be69d6106a48548e7626207dddd7042a418dbf372cefd05e0cdba61b6",  # pruned original safetensors
                    "e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053",  # ema pruned original ckpt
                    "cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516",  # pruned ema original ckpt
                    "19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1",  # diffusers safetensors
                    "cd1b6db09a81cb1d39fbd245a89c1e3db9da9fe8eba5e8f9098ea6c4994221d3",  # diffusers non ema safetensors
                    "c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21",  # diffusers fp16
                ],
                "layer_b3": [
                    "909c6ff3192ab2767e789a6125865bc23163db467ab78b1c633bad46a4293fad",
                    "b52807536902cabbf84f99e4fa2f8713fb4ef77e739f06367ee0d486e3222faa",  # ckpt
                    "d31382d71a1044b636d80d861a2b4dbca51826bed34d34b5c14608b7679ccefd",  # safetensors ema pruned
                    "5fd8b28013b7e5a64c7c235f0a93d93e48bc19a0e5dde7b646a87b429219643a",  # safetensors pruned
                    "731f552f29edcb4f86112cc94d296377f3533a9633ccf83e202d9e1785d94a00",  # diffusers
                    "2d2f97574a161cf01a6f6d476b141c7be06f940d94b695ffc12c4e74eca2de1c",  # diffusers fp16
                ],
                "layer_256": [
                    "ece771354ad470a82d56eda413ae3dd6c00d2de28ab3c56a88201d08d4424b4b",
                    "65b084dada803461ab9ca9be9b892d211870a121dd6c555a111eea470b951c54",  # st
                    "dc937b59892604f5a86ac96936cd7ff09e25f18ae6b758e8014a24c7fa039e91",  # ckpt
                    "92565dec90f7c8412dc872e820f66cd0c56263bbbc392439645b6fee270f41bb",  # st fp16
                ],
            },
        ),
        (
            "Kwai-Kolors/Kolors-diffusers",
            "KolorsPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.float.F16",
                        "generation": {
                            "negative_prompt": "",
                            "guidance_scale": 5.0,
                            "num_inference_steps": 50,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                    1: {"diffusers": "DiffusionPipeline"},
                },
                "file_256": [
                    "425ff1dcbe3a70ac13d3afdd69bd4e3176b0c3260722527c80b210f11d2d966c",  # fp16,
                ],
                "layer_b3": [
                    "6eb15506fa38b4cbb26391ab1b6c9ead05f86c711e46583bfbe8fc4421571414",  # fp16
                ],
                "layer_256": [
                    "04e3c17170b8a200481f6941b370fdc5056a00fe5a16956de01790f8a93c0dcd",  # fp16
                ],
                "identifiers": [".DenseReluDense.wi.weight", "encoder_hid_proj.weight"],
            },
        ),
        (
            "stabilityai/stable-cascade-prior",
            "StableCascadePriorPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.bfloat.B16",
                        "generation": {
                            "negative_prompt": "",
                            "num_images_per_prompt": 1,
                            "num_inference_steps": 20,
                            "guidance_scale": 4.0,
                            "width": 1024,
                            "height": 1024,
                        },
                    }
                },
                "file_256": [
                    "673b3173b037fb5f65b14fde37267390641a36726683de75dcf9df76fce2b866",  # lite bf16
                    "45c1eb5ce9b69efac891ad459b15c215cd90a986adbbfaf3effd3a89578cbcaf",  # pretrained
                    "088ddf1e444abf399007b2da2bac87791df165c69f477994f6b3c745a20904b0",  # stage c modelspec sai
                    "39cec96c7212607f9e526db719bf1df507166d09f4748676c13b0d31cd4adb07",  # stage c
                    "31ffe2f1a3e2351d658fc7d3002a4eca22466a680f7fb3715b1e3768476f9633",  # stage c lite
                    "dfe24009fc881011f350d08d9d13be13a1a3b3cbfed667435efe0fd419aca099",  # bf16
                ],
                "layer_b3": [
                    "c55c83fa435ed128457f605bf1312e54727996d1c94413fc5ab5b49e9933857c",
                    "6fb07ed9fc6ee636e50783802754b3a37bbecfc67037813b616223aeaf6fe877",
                    "2ea194240e105c8962923e2baca88cb6a0c826794afc2ef82474301694711d68",
                    "3412c8a184805621e4595d57268ced0b5c3c1974cd221bf67b2c908eec4fd61c",
                    "53abfb013cfb0e41d0bc7b96bb83e42a4d4c67cb7325f9acf645b02d90efd8fe",
                    "34556558f680c183adc2accd493cb9888a98ba853226bbecb07d95eb2055ff4f",
                ],
                "layer_256": [
                    "4f5e0a738b963d3d4f8413387a0966ac1ce51f0f985bcbcc124fa221a2fff467",
                    "8aa77e732a398b7d0dcd9a35d5682c2b5ab090ae90e915c7c91878abff0284d8",
                    "4bbd46ded0916de3108f0da7145a80f5c7acea26ed35b0aaa29af12008352453",
                    "415d1f3ecd06416708c1b83ab21e50b39c9d88d19dc33e60b977b7b7061880b9",
                    "f678c32815c238e14091f690c8a83c3375c8f7738dc7abff79ff086ed9b59204",
                    "17c8da803df7b9bbc8b1d7cc0c44916fea5b5ac0891330c4fdf0326fcd4496cb",
                ],
                "identifiers": ["down_blocks.0.2.kv_mapper", "previewer", "backbone"],
            },
        ),
        (
            "black-forest-labs/FLUX.1-dev",
            "FluxPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.bfloat.B16",
                        "generation": {
                            "height": 1024,
                            "width": 1024,
                            "guidance_scale": 3.5,
                            "num_inference_steps": 50,
                            "max_sequence_length": 512,
                        },
                    },
                    1: {
                        "mflux": {"Flux1": {"model_name": "dev"}},
                        "generation": {
                            "height": 1024,
                            "width": 1024,
                            "gudance": 3.5,
                            "steps": 25,
                        },
                    },
                },
                "file_256": [
                    "f6315581b7cddd450b9aba72b4e9ccf8b6580dc1a6b9538aff43ee26a1a3b6c2",  # krea sai
                    "1b2170ac37156d4cf91909eb6834bb8adac84bc1fce8098a29cfb03738df84ad",  # krea diffusers
                    "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7",  # modelspec sai
                    "d86a3038eacaa720682cb9b1da3c49fecf8a3ded605af4def6061eaa18903eb8",  # diffusers
                    "b7d840eef01c27dfd72ae9143c261355a51bab3b2662263a6cb0059d55347c3d",  # qwen2
                ],
                "layer_b3": [
                    "261559c8eaccae558f72621804a9ee188d338e45e2c622a58db709ac190198ba",
                    "87f5d565c66e40eb02eb96498243ad81afcbf86192db99a4fc8fff215470320e",  # modelspec sai
                    "e61d10a394902dadca9367467b2245070f651f4553ec4a96192fbba64e820acb",  # diffusers
                ],
                "layer_256": [
                    "3db58cf834d2f81abb1e035131956da4c90451074c681d0db10810e55e60c2c4",
                    "ddf1a34a06b355ce2bcd0f9beb0713450d9bcdc61a03a6bc37716361735e96f1",  # diffusers
                    "ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881",  # modelspec sai
                ],
                "identifiers": [
                    "double_blocks.12.txt_mod.lin.weight",
                    "add_q_proj.weight",
                    "single_transformer_blocks.9.norm.linear.weight",
                ],
            },
        ),
        (
            "black-forest-labs/FLUX.1-schnell",
            "FluxPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.bfloat.B16",
                        "generation": {
                            "height": 1024,
                            "width": 1024,
                            "guidance_scale": 0.0,
                            "num_inference_steps": 4,
                            "max_sequence_length": 256,
                        },
                    },
                    1: {
                        "mflux": {"Flux1": {"model_name": "schnell"}},
                        "generation": {
                            "height": 1024,
                            "width": 1024,
                            "steps": 4,
                        },
                    },
                },
                "identifiers": [
                    "double_blocks.12.txt_mod.lin.weight",
                    "add_q_proj.weight",
                    "single_transformer_blocks.9.norm.linear.weight",
                ],
                "file_256": [
                    "9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a",  # sai modelspec
                    "9b633dbe87316385c5b1c262bd4b5a01e3d955170661d63dcec8a01e89c0d820",  # diffusers
                ],
                "layer_b3": [
                    "c65ba812ce3ce056eb1585673f62fb896afe6ec049faaf00a97bc35c9a398c44",
                    "03049273329fc7db2da10de6d3eb27cb03f190e379c0556cc97b3f0f29001d0c",  # sai modelspec
                    "483c4be8ef031c56bc8450d1a3cfbe54445ed317bcd801be5abe89f1d3c48790",  # diffusers
                ],
                "layer_256": [
                    "79c07e339865fe9e22c80f723d728c778130acd07a330339c68218b92bb7b3b8",
                    "ef5c9cd1ebe6e3be5e8b1347eca0a6f0b138986c71220a7f1c2c14f29d01beed",  # sai modelspec
                    "27bc71eca2d2ff7459165acc12010230911db7709a4f6a5c255befedfa6b1649",  # diffusers
                ],
            },
        ),
        (
            "stabilityai/stable-cascade",
            "StableCascadeDecoderPipeline",
            {
                "pkg": {  # prior=prior_unet
                    0: {
                        "generation": {  # image_embeddings=prior_output.image_embeddings,
                            "negative_prompt": "",
                            "guidance_scale": 0.0,
                            "output_type": "pil",
                            "num_inference_steps": 10,
                        },
                        "precision": "ops.precision.bfloat.B16",
                    },
                },
                "file_256": [
                    "fe92687deefcfb33bb3ec181254b55fe4e434c5084ce9d38815eaa32487ad376",  # lite bf16
                    "2c8d58b267678aecfa6705a0a0375c88613065a8a8d32ad3a4c3867f5461cb3a",  # bf16
                    "6c218dc948575e3b14b03dffe2014d7870ac505005770ce3abdc28e920a03c05",  # b modelspec sai
                    "a6c3d534a9be308e95d2c3224af94a854bebd9b503f620f1ae3c8e6ba4a341bf",  # lite
                    "7b431ea7d0f10e72b3eaece353bf6bf2f6bc717b6f4207411be186b40dec1f43",  # b
                ],
                "layer_b3": [
                    "9506d989de0226018de214f7ced4670eb5aad4a0c399a9229488ceccdf9a3ceb",
                    "6c09dcb83e0cd7ad735eb763c5e3721c579d796853f0b9d31ba74fb13cad4f94",
                    "e07025965cee925e31f1d617ea8baa575e7db910d40cc0482fd83df317c0812b",
                    "d9a42e4226fb2778aaeaf0d6bda173a4ff95aa574c6d9e27e41542aa469e40a3",
                    "8dcd87dc7a9b877e8e2a00abac44c4da9eadf2b8df4ae68f27415bb791381a96",
                ],
                "layer_256": [
                    "630ec0f3adf97145316c034139836f9df952060d0237ac4e478c55d9a3a50bc8",
                    "80904f707c192ddd06be2cebeb2ebbec3eb0e9c99076d50824d391ef3ac67bf2",
                    "8ccedbe1e8cc4093f05b5f8d90e6103e688ae1ac71e0d6261fb17c42ff7c25e4",
                    "3524e7fa9ca6f7ef695bc2d3410934eabd5272946a05c8cacd7f329e0bd9f1dd",
                    "40499a8f45ae28558ed2fe4fc549a4cb469bd237434b331ccc0b1910310ed733",
                ],
                "identifiers": ["0.2.channelwise", "clip_mapper.bias", ".12.self_attn.k_proj.weight"],
            },
        ),
        (
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "WanImageToVideoPipeline",
            {
                "file_256": [
                    "b4602c35fa0519750a42c03e3f296c02d542291e344c4d702522cddbd1711f13",  # 480 diffusers
                    "6d7a34b63b70eb608324e546d979167a5e787ac6bca3528e63f54a11572d66aa",  # 720 fp8 scaled sai
                    "b2051cd29d6b2f0c924fa7a3e78a4772f0134d7b059f21590dcce416f4f6cbe8",  # 720 fp8 sai
                    "7664fe075b3c82dcecf89012ad3429eee41ee9f10d476f60bc2d2ae3c4ca986c",  # 720 fp16 sai
                    "8ef7ea5bf9eea636b9b3ebd84c40671b4a18ae2704cb4c8595cb5b25c1d8e8b9",  # 720 bf16 sai
                    "b2de21b99b2e72cb0ff15253b07e926f26e7cf1b7e229efc32f94ad1f1ed9395",  # 480 fp8e4m scaled sai
                    "0ca75338e7a47ca7cacddb7e626647e65829c497387f718ecb6ea0bae456944a",  # 480 fp8 scaled
                    "c058a4ac5363c35d1ab4dd3bdec788c23b267fa42a0d7c68aba599f2f74600c9",  # 480 bf16 sai
                    "27988f6b510eb8d5fdd7485671b54897f8683f2bba7a772c5671be21d3491253",  # 480 fp16 sai
                ],
                "layer_b3": [
                    "4b6c3354c9ee5694e00a78f5658fdf14129f159c3b78a57f82fb18e0f265a83d",
                    "c36c783559a40d22504f6c4bfb4f5aae760f3f46bbb3a595be79880935122175",  # fp8 scaled
                    "ac62f7d5583fd2e85b738fafaf233e2cde6e2857e04351135bb9ded45f9082ce",  # fp8
                    "215e89e855b5e9456af9aa68bc67567dc2269002aaa6b01d849ffec425fc628d",  # fp16
                    "324b8b6c2d512547a2c31bafa12e20acf313fd3aad587b293334f9f629edeec6",  # bf16
                ],
                "layer_256": [
                    "137881dad8c00063bc8bf05f93067736e419173cd171acc22f77b730db688a19",
                    "8c5952fd3d333d3a4b719bf7d8ce6b12d1d2e78caaa7e42d713788cfdcadd244",  # fp8 scaled
                    "86c58bc4864c97f394ea6bccb2ecedc4aab7166f5b9bfeb313edfdcb2918164a",
                    "cac45f7d8f1a0628cb0738bd308689e439b1cc6206e5f887d60d5b37d30138f2",
                    "60e4f71a0961b1346b6f6b5ebe4c8cc93219239c5e13b4c0f1e19e9b8e1324d5",
                ],
            },
        ),
        (
            "Qwen/Qwen-Image",
            "QwenImagePipeline",
            {
                "file_256": [
                    "9f33a59093af3abcc2836d4cf4b7bd122c238ca70a26c70f34fdde64646b3bcd",
                ],
                "layer_b3": [
                    "c87eedda853c12844a8deb3592a90bbcbd4dff2f7a850c28755e4aa171432150",  # diffusers
                ],
                "layer_256": [
                    "fda2472d8ef6587a4c979021a2390eeb7c8fc2bcf565330ab8dc6b22f5348ec9",  # diffusers
                ],
            },
        ),
        (
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "WanVACEPipeline",
            {
                "file_256": [
                    "bd8bbb8834a274525ab65cbb063f21aa58973a054bfd1638bfe395504c9d9b99",  # diffusers 14
                    "192804a4e10b5bb0a13f5c224bc4ec9707b3b8cc0def8eea005dbce7c9d6752a",  # diffusers 1.3
                    "f202a5c59b8a91ada1862c46a038214f1f7f216c61ec8350d25f69b919da4307",  # 14 fp16 sai
                    "654693bf2a93a27cd67c3bcee238bc1d0cbb0dd9a74928ed7155fb21a2a1900a",  # 1.3 preview fp16 sai
                    "640ccc0577e6a5d4bb15cd91b11b699ef914fc55f126c5a1c544e152130784f2",  # 1.3 fp16 sai
                ],
                "layer_b3": [
                    "5357d78799a61cd2d72a8a2824c919d63f718eb3fba624af63689e9c657db032",  # diffusers 14
                    "7ae67b7ccf79d1c3f4531ae138e1eb63d52dd97a66b3fcbe1d68fded8df4d5b1",  # diffusers 1.3
                    "ee63ecdfb3da6901853a59ec950f3e7c3f6595ac46347a03881a4a9c71425377",  # 14 fp16 sai
                    "82762df3539021d3c0342e0da04137ddbe95ef37ea933cd0a68c09c2c650f2ac",  # 1.3 fp16 sai
                ],
                "layer_256": [
                    "2684413479030170fb3f08c1069c02957ffc386a59168d23b55d579d5c675269",  # diffusers 14
                    "d527680fa735e5f30ef8852aabf8a49f02a094bc4718f0787c5b85710a13c026",  # diffusers 1.3
                    "9677492a107b3ed827c7285db3393f5321d451cc6d922a4d0488d2a67e939446",  # 14 fp16 sai
                    "aaef66a4f65ecf852888d160b2122753fe4c6d642b5d41db29e4ce9e6855b5a0",  # 1.3 fp16 sai
                ],
            },
        ),
        (
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "WanPipeline",
            {
                "pkg": {
                    0: {
                        "diffusers": "WanPipeline",
                        "precision": "ops.precision.bfloat.B16",
                        "generation": {
                            "height": 480,
                            "width": 832,
                            "num_frames": 81,
                            "guidance_scale": 5.0,
                        },
                    },
                },
                "file_256": [
                    "299e6304544f2783896372fa919e755a8bb9ab8caf898ce08a678dae391e1179",  # diffusers
                    "a9278e6e9c82d174e6c67b3c97d8b97fef30af51dcf59160f2fc241f6819f5dc",  # diffusers 2
                    "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",  # fp16 sai 1.3
                    "6f999b0d6cb9a72b3d98ac386ed96f57f8cecae13994a69232514ea4974ad5fd",  # bf16 sai 1.3
                    "2e39adde59c5e0e90edbb35873126b0d67928b5c11c501e384e976d6dc597cce",  # fp8 scaled sai
                    "2ee88ab18d7ed7691c5b7f8bdc3d0a9815e6efe75499287564830fd209d3cdfb",  # fp8 sai
                    "46c27d3693bf2475990a912e08bf67fc6e6cd5396eab87b5e8dd1fcd3651364a",  # fp16 sai
                    "193535c6450045f718df5f011de6d94d49bd9b13f37ca0412500f050dbbb01a8",  # bf16 sai
                ],
                "layer_b3": [
                    "32266d1c79b518adb9d21837e6a427f6ae55b68cfdd673a7dadb38820fddeb48",  # diff
                    "3b6989856f4f05368524c1852d8660b73c84cfbe44460af017d7139c2a4641b8",  # fp16 sai 1.3
                    "f4d6cee3c112db93b3c9137ad102ec0e79ec7ab68b9bbc59004fbc268ccd5ddb",  # bf16 sai
                    "e627144f41055619eb5407699c46e69ac0d87cf8873721e3e48c9e842656abf8",  # fp8 scaled sai
                    "6c00f3fadedacb841c4b9b4321b94a11ef85a08c9dd9253e5f9ba95856715579",  # fp8 sai
                    "a0c339253c714b05877c8fbab649ed631cf021930978f3696a46f685a07c9092",  # fp16 sai
                    "6435da89a870fd0e88680d31de75b9a40c408a4768eff384ce9b9e99481e8e66",
                ],
                "layer_256": [
                    "52493c23c5fc1d087a283bc4eabb151421b7ae09affa12a5bb059d62656c5766",
                    "058dedb3d2683a9a5b671c6302690e22722c93f6ed92281d5fa74ab190e632a1",
                    "5fbed4b95e7196d3626003ea9e0fbbffd074b4297ca406e01b5b6c5d881a6080",
                    "3a2335c8e7a4359c071b50333b5c00eef6f42a1d5206915e2ee99464a8c5eae7",
                    "0542780670dd75d4cd9deda123d2e150730646c0a1a8d34582460991498a77a6",
                    "e925b8222774905c8fbf10af77811fde7870e563eedcde2c94bd5c727e952d49",
                    "3d915854976284347efa7aa0a117c0fc3b415c4208e1a6c94beb4ccb9720743d",
                ],
            },
        ),
        (
            "nvidia/cosmos-predict2-text2image",
            "Cosmos2TextToImagePipeline",
            {
                "file_256": [
                    "7fbd20dae97cc26a55c7aff3024bc84e554cff8f69966c725a24c8238c5431ec",  # gguf
                    "6d211f1c14cd793156da3a840dd5462ae072046fcd6f1dc64c613a5343bfe896",
                    "95a2b32ad31a271eb64d35985c7ea46f1448528af70932eb1f35d57f90c27be2",
                    "344e67faf333b7849fa94290c9028bdd5e40eb19700754c833cda0423bc10ad0",
                    "ce15ef565cbb9ef414a6f7a396c455d82d5f762d2174493da87fe009c5fee75b",
                    "94aa9f2b59330b88e97b6b439e2f206a51c86e6b154fb66d43ed149bfac23cf8",
                    "636de5388da249130d51752991a1792b90af31cbf43f021ae07f75756ee2d79a",
                    "472c5e4cf5056a1a59085addb5a86d801de39bf5e000d253f206a7f63c710029",
                    "663266ace67c22529c3b6bfa0e8bd69f0ba6e683f5f02b8e3da50881057ba142",
                    "21a674b314c1364d0dbb3712f5ed702996a7b7403c452835cac22709e01c2f77",
                    "3bf2df806c6472e039efc9e8d3181163d7faa7b385e61519b7d17d5e9c993a49",
                    "1de35e1603c4c30bc80b132ccea15fc0503369caf68290708f17e679e98cd41f",
                    "0738e559bbd71f7351ccba34b2b47362a3f829b92f3dbcffeaf1e44b0d52f42c",
                ],
                "layer_b3": [
                    "5a18ba14c41c6601dcc1195ca180ac7744357eb15ace39272788bda1a7151e9b",  # gguf
                    "67cc3eaf7987c89cd7ccff13de6bc03e3eec59d260d44486e2367cd946ce6f20",
                    "3c6fefa107742488d2e6856714198a762f2fd35c67edd50d4657eaf4b59c7ca3",
                    "4e1f90ee1e8959d334c9b1ea2cc5e58d0b8340e271c35f81c8a5ec26e16d9d76",
                    "f8171071e828524fcc2806126ad100a2198e450c82c0864c8fe8b358c5cbbfbd",
                    "8126101a0207ecfbd741394fd59f306bcb4c492b2a921e0921c426ca7bd38985",
                    "c942c5a85ff7cb602d8ca894f5d180c2224e91f0b62c3a21f6a425f9e0e8554b",
                    "c8c500de74da879a547875fe1046f62ab18bdfd09c09eb3da723cbc2319cb4e3",
                    "c0ac3f67501004e9e9a55d1658402ad97e42bf8a266edf81f6f3bb835ee476b9",
                    "84f5926eb4e11d826815682b076ed7d3bba4c86520859be80aa1ef92c72b26a4",
                    "1d4375aab5548708559b0fde150754a2163cd211eb20a5471e17afaeeb26e082",
                    "68bd8982f59c60d69c301d16dfb5a60f5d43d66c0b60138d48a22f5ded598e7b",
                    "c3e9a10cad7aebf979072092008be6e2815d03d28cbf316c15e8daf22116bd7d",
                ],
                "layer_256": [
                    "38f2a75eab667c0cc85f3946a23ca6dc2278438c25a9f93aaaa9f79c3808e180",  # gguf
                    "ee8434a5e9bc6fa07199de2d0c69fb87f7922c31792bafd13f527c9d92fecb0c",
                    "2f8382657babb4d0ae4f8e425ae33b21ad71deb6ba457fd6734f05208d52e06a",
                    "34b181a8291b571857cdbf67ac0081fea594a2f223bf20bd2fc8b0c889e9602d",
                    "d198c412b972e381acfb812304fa98ed0d97a2f072ddc195cd9a1eb83b1d8146",
                    "79580a13aff9859e67b0a9f4f8893236cdcfa58c3d43770641aaac8daee55a94",
                    "cfd48c7ad71c913fa8768167ed0c2ee8c207311b22b1e5a8761369b5a780e8d6",
                    "da91362ad85d4d2e80a2cb7a55e4ae0e52c9eef8b437a95894ce5ab75d36568c",
                    "15f84001f5205b6dd8c6f1334cb51c46f6171c7795fb2a557ea16b874f0c71e5",
                    "5d29179ad15a15d2561defcdda66f1d1e4d065c1e0738f9cba4db5b68b93d2ea",
                    "7ec489d1e461f5fb2af627b68034ca57f19c516aeccbc5d188b3bd27e3353a15",
                    "c8dc42fe7b411d746ebdf86286b91cd6893c5f028076b8fe4103f7ea8e1d8833",
                    "86df7c095aee01588e961438f322b85ca0100a9e440b8a2b6c724e00f748d8b5",
                ],
            },
        ),
        (
            "rhymes-ai/Allegro",
            "AllegroPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.bfloat.B16",
                        "generation": {
                            "guidance_scale": 7.5,
                            "max_sequence_length": 512,
                            "num_inference_steps": 100,
                        },
                    },
                },
                "file_256": ["6927dcc812841c1da549bf11c97ddf30532aee0e708a6642fa64cf8e0dfcdef7"],
                "layer_b3": ["8b20714a6af89ea4bf4ada1f805c5b9d529ef136c229e9b75392242d62d80c3e"],
                "layer_256": ["9e44e6c919dc71c24a193641e6265cd9983a2a773b9bbaf527c10ac4837b29fd"],
            },
        ),
        (
            "audioldm-s-v2",
            "AudioLDMPipeline",
            {
                "file_256": ["fc30d5b5a3bb8d08672736efb1fff10755ba7024dace39b2dcb579a105aa2a5a"],
                "layer_b3": ["82fbcc553c1ad770d28fd1866b935249c5ebfbf75f3166ae823e1bc6ef39a95a"],
                "layer_256": ["d076446a58a36bf436e37444679d62bcf2f45689d4aa3d799b3fe801c71ed2c8"],
            },
        ),
        (
            "zai-org/CogVideoX-2b",
            "CogVideoXPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.float.F16",
                        "generation": {"num_videos_per_prompt": 1, "num_inference_steps": 50, "num_frames": 49, "guidance_scale": 6},
                    }
                },
                "file_256": ["8fbb6a5e67c70885a8ed8e33df144ac61253e45977be5035fa18cfdf77d386c7"],
                "layer_b3": ["1db3439649b5362448455fb2ed6ebde0c3b973655a206832731149757ad165bb"],
                "layer_256": ["edd6bd51f1236f528ff8d32dc754f0b86cfac901b800642ea497358156dc00bd"],
            },
        ),
        (
            "HiDream-ai/HiDream-I1-Full",
            "StableDiffusion3Pipeline",
            {
                "file_256": ["3cb3f6d77a3fce19b90fa7f66da0cbe997b0785a38a788b559290d3062f6fd26"],
                "layer_b3": ["612eb9b2676a3e7b28b10aae045a97a95de2a399fe3801c8f6369589c3a832a6"],
                "layer_256": ["78fbfb7fddb9ccbdf91f22b0c3d304cbf0cc7305dbccb216982233849ec727df"],
            },
        ),
        (
            "cvssp/audioldm2",
            "AudioLDM2Pipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.float.F16",
                        "generation": {"num_inference_steps": 200, "audio_length_in_s": 10.0},
                    },
                },
                "file_256": ["359a5ffb89a844beb2fcfac584aae2cd7cd6e87c3ab1ec4e892ef45d91db77c2"],
                "layer_b3": ["eac241273f9f30982fc04aa88b4dc1c38b533430956a55b9ed4d3e5c717ec962"],
                "layer_256": ["ab109d01b43788063802f00c6ecab024c830ea58d668f5c2df9e3ae5b87d86cb"],
            },
        ),
        (
            "tencent-hunyuan/hunyuandiT-v1.2-diffusers",
            "HunyuanDiTPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.float.F16",
                    }
                },
                "file_256": ["7d31ac8fa389ff39dd0a81430010e52c43b59f15adc00c83625a47881e16830e"],
                "layer_b3": ["bccd37ecc9f85d132b46d0bb67b4facb49fc6c091428a4feba9ab9a93140f5fe"],
                "layer_256": ["ed25d241d58ca298d28abd5919e70341ad194e77dce4859436b52ea4d8fcb616"],
            },
        ),
        (
            "Alpha-VLLM/Lumina-Image-2.0",
            "Lumina2Pipeline",
            {
                "pkg": {},
                "file_256": [
                    "132b4d213fdd3cfc14333746fc3eb8bbe6358cd73c3bc95ac4ccec230b97dca3",
                    "a7c09ebae62996a8289782161338a3cdba58c11d2d849c50b2d6502e152b0d6d",  # pth single file
                ],
                "layer_b3": [
                    "198bde52f09736f1fc650dcdbd0e6b0f6a5ce186582554c1d9ee8ab16ac0feb2",
                    "b52807536902cabbf84f99e4fa2f8713fb4ef77e739f06367ee0d486e3222faa",
                ],
                "layer_256": [
                    "982893c99860aac8198c2e435cf85f782fce8f10732daf1f2881a26864400a4e",
                    "dc937b59892604f5a86ac96936cd7ff09e25f18ae6b758e8014a24c7fa039e91",
                ],
            },
        ),
        (
            "ucsd-reach/musicldm",
            "MusicLDMPipeline",
            {
                "pkg": {
                    0: {
                        "generation": {
                            "num_inference_steps": 200,
                            "audio_length_in_s": 10.0,
                        },
                    }
                },
                "file_256": [
                    "853d0ef1d61cbf5d682872322ea8b761ba3d2f85bfbccd58363bd6b2f837268f",  #
                ],
                "layer_b3": [
                    "82fbcc553c1ad770d28fd1866b935249c5ebfbf75f3166ae823e1bc6ef39a95a"  #
                ],
                "layer_256": [
                    "d076446a58a36bf436e37444679d62bcf2f45689d4aa3d799b3fe801c71ed2c8",  #
                ],
            },
        ),
        (
            "openai/shap-e",
            "ShapEPipeline",
            {
                "pkg": {
                    0: {
                        "precision": "ops.precision.float.F16",
                        "generation": {"num_inference_steps": 64, "size": 256, "guidance_scale": 15},
                    }
                },
            },
        ),
        (
            "hunyuanvideo-community/HunyuanVideo",
            "HunyuanVideoFramepackPipeline",
            {
                "file_256": [
                    "bdb957b35585ea74ae42ca92865a68fa1bf1ebc6c5b7e686a889e5c977dc24c7",  #
                ],
                "layer_b3": [
                    "d31c56b4c9444d4c2f1b10120fe964e0956f6b8c7e7c1e4cc5a1f37406fc49f5"  #
                ],
                "layer_256": [
                    "fe741fdfd163bcb1e0ed81d80f79ac3576dbf6e6740674efadfeff782a48bed4",  #
                ],
            },
        ),
    ]

    transformer_details = [
        (
            "google-t5/t5-small",
            "T5Model",
            {
                "identifiers": [
                    [4096],
                    "encoder.embed_tokens.weight",
                    "text_encoders.t5xxl.transformer.shared.weight",
                    "t5xxl",
                    "encoder.block.0.layer.1.DenseReluDense.wi.weight",  # small\
                ],
                "file_256": [
                    "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e",  # shuttle/flux diffusers# flux dev
                    "565cb2487351282e8e4dbeb88e63f4ad28217ce0439f5a8e6525a924807d2d9b",  # bf16 modelspec sai
                    "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",  # fp16 modelspec sai
                    "4f2751ceeb2a96edd693e539dc5d6bba0b8d3814f49a9b3798403a0cec4b2e3d",  # fp16 diffusers  cogvideox
                    "83690f3cc37cecb5e907f41ab0f7abb0855ef24a0a8aab9259f2888ce85a34e2",  # flux diffusers
                    "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",  # fp8_e4m3fn
                    "8490f7a22615c20651a63dbe7b4241929826a4de20292dc8e63bfc3c61e3654f",  # qfp8_e4m34n
                    "d8720addef2596fef86b1b22e4b62875c9118779ba8723759a75dfcbc649ffd5",  # mystic mlx
                    "7d0eac95abe8daae454bcd3d166b8bfc6a35fe68278f97479d62dbb6850f38c0",  # mlx flex2
                    "ceabd6f71c7112cfaa4dfca8711dda97b79fb9b25983f1c95532de226045f1f8",  # mlx jaguar q8
                    "49e139f50824fef40908ef4307c851e7adaa8b91bed44054c4829600dbedfdda",  # mlx shuttle 3 q4
                    "211ade1d474f5dc83190aec8be5c4baf52643777790d64de0cbd84f63613e5e9",  # mlx flex1 q8
                    "7894547154ba3fd6e364e66e2951ee82b4c3fc1ae0f95df6a4f9d1c5a4e98f17",  # DeepFloyd/t5-v1_1-xxl sft
                    "eb529f693f4b17773a24e787fcba29486d5e1700dadcc20bb91e4c8b00212d08",  # pixart a
                    "d80116f6fc39801e4eef425a584e7a7a41cbe5119797bef2dad67299909fe2ae",  # Q6K
                    "31ebe18e901bfb6e5709a20ec1c95fce29bce2b9545073231e0f909a53239f5c",  # Q3 KS
                    "6be2b0b7e2de7cf2919340c88cb802a103a997ce46c53131cec91958c1db1af4",  # Q4 KM
                    "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",  # q5 k m gguf
                    "9ec60f6028534b7fe5af439fcb535d75a68592a9ca3fcdeb175ef89e3ee99825",  # q8 0
                    "8f5ab879234384235d56732f0cda07bf8801f30a49645248c5bfdeeb1665f64b",  # q3 kl
                    "86427a1f4dba48940e45bf78d6db5bf0d48fce8b4656f5aba27955f06af9628e",  # q5ks
                    "88b696cfae098f03bb078cc5944ef03aec1e91ec020a6b016b723a0f0532558c",  # q4ks
                    "1dc600961d3c5ed081f6700485cdc7ed9cfb4631f2dc385b7ac6bd3c80846d0d",  # f16 gguf
                    "f28631189911f8d7931e8fe642a4cb2a3c51f50da7cabbfa06b89bafc19c00d0",  # q3km
                    "de9dfdd19d7ba6859993cadec5100665dc7a4fb71e1c6c8970959cbdaf4366e3",  # f32gguf
                    "7a68b2c8c080696a10109612a649bc69330991ecfea65930ccfdfbdb011f2686",  # allegro
                    "2c0c539ab8e8fba3877cc94bc483e427f74c525f817a809b028ebc8d96d75a94",  # hyd 1.1
                ],
                "layer_b3": [
                    "ca94e03b7b1fdcb0d6ff5205eac56f145d2dff8a9c489faf80935bfec8387f18",  # bf16
                    "c0e2b054bedd782909191b05748a88c28d1538fa91789fec63f036ba01dcc001",  # fp16 sd35
                    "672de9b79d14001de7d1109ffc52e4d0cccc3bfee6f45648fa347703b58e2b99",  # fp16 sd35 diffusers
                    "abdb187a996c51cb0469630c124b14eeb0bb8f5f635aca6c71dea264f8bd61ae",  # shuttle 3 aesthetic diffusers
                    "8926f862b7763fd9688af317eba7809aa71a478484be0c738c269de368ace4a7",  # diffusers
                    "e616b754cf55e55b3f9f17ab7e1fff95f0607c81782822fc1223ae22fb1e9f36",  # fp8 e4m3fn
                    "b79e5f1878a62cd726bb4f9fc1415cacb071d278440e9026290c7b36cb41e1d4",  # fp8 e4m3fn sd35
                    "77619d5278d9f547ddac17d4d99df56cb6a3a9e660ae31b2f896a4297907e62e",  # mlx t5 jaguar
                    "c87c9d3cc7becc46ee34821299cf8551a6df5541582a45469a031bccdc4bd340",  # mlx shuttle t5 q8
                    "7e6c32c01c89fc5d1610c410135aa9708e77a7444510e5e479fa677ff2b53643",  # mlx jaguar q8
                    "a49c2bc301733967ddff113790e301773dc5dd71368b657af4141458de593ced",  # mlx flex2 preview
                    "c2ea94030ea362e03d73d448fa5353ace0a449dc38c51a4a49fb148444ebb8ef",  # mlx shuttle3 diff q4
                    "4a90463350f08ef41479da1d561ab41b8f8b792f1603a092226a838156aebfb0",  # mlx flex1 alpha q8
                    "f86cd0324eebbffb81b15ad47dc8b63fedfa51dc222e44e1a958a7becce2bcb0",  # df safetensors
                    "48c54c61c5f14e42761c6177539b2da3a22222516dab053952ca8d8e92f93d65",  # pixart a
                    "311332d9738773669128814d944b1e860a8e3176b37abf43370bc06b43b454d0",  # flux
                    "3f4e51dec6d542759cdea49b3bec14c090a4908f953fa3e182e2ea43b5b05402",  #  q5 k m gguf
                    "beb25461e168359108add77263ea5cc121b7584cc4aa304ffc4e134783bb1d88",  # ggufs
                    "43313f90a359c8c1c787a7a833b1ab9f7a38204ba36d0ba587c658d0d9bf0852",
                    "fa9e97cdad26f55fedab83a3f114e0338c9cca3ea2bf8f1b168a6dfc5919bf8e",
                    "93108d67f8829a7e1e8f3773e9ce53c67f365889c2acfd69816ac80fd43f8e08",
                    "fc65a6cc55e89394d7bc0fa4ee952d63ce3bdc143b84b5aa4bb3edf7722a6b83",
                    "8163bc781a7e013dfeb806bbb828a36913cf119363ea5fcd9071d87a0c227cda",
                    "ad2ba63e1134bad1b15ee339313bc130708b2995e8b4b76fb44d727f28c26ad9",
                    "4a844772638ffed2f61d45eaac984094b92540fa1391a4098608fc73a6cd4fd8",
                    "76c31e1fd35da7de7cee97c1e7c5ccde640e6fac3e17a62e115ecf484c7196c3",
                    "a4d672e22b5bdd8f8b0885cec4a173d0466bb1dcbfbf8400cedcc41c2494f16c",  # ggufs
                    "d1860c3f01dc9f260d98b50d3d2bbc8dc2d3eefaa93778a8de9d7adfb897fc6e",  # allegro
                    "b8719092fc58487406211f52dc55bf40b573ccfd29933a989c33a36b694f6f0a",  # cogvideox
                    "795e272409bc4fa55f402485acf86b607256f91aa965295c5bb771c61f8e9e74",  # hyd 1.1
                ],
                "layer_256": [
                    "bb20f7805209379aea4d6548f17e551cf27d0f8426ca169e4df8234f718ed5ef",
                    "431580c2d86f9a9ed3500f776a4c997223e5644aed211f965354869ccfa4d76e",
                    "2ccd548c4ffe34168c60779ebd497b9b410981a2fda813c8723a24a805c94ea0",
                    "a608fc4e1cc9762e46187a1ce66e98e8ba4bc3a604cbfd96174bd876baea0fa1",
                    "dc9e74cdf535e0b7a17e1335d0d8b38a00f94facf0cb01363baee09945a25278",
                    "f07409710a69b2247aa4723a9b40d2225d5e5bfba7b60c51f0ea901fc2ef5ad9",
                    "ed28f8b6cc472f352fc840b5a9f841ff17d76ae6918f0676464dca20529aa92b",
                    "97c1a08f87c59b4c55ad4672841977cfce43ca7730bcd11d8c178a9330de1855",
                    "968972839b859a9c4457f190fad2e17e8585ce27d9ef318df4f5b4e902143944",
                    "4dbdeadc957c898c327197a3d8770188535672e9208beb29bbf48dfdf51c8955",
                    "669172c2b5e8b97774d9dd0227ede40c4d25cae3adae97d9f281d03531e7e137",
                    "39fff130b9ee240102c28a78ee1c4a643e9f800b734ff133f3ab2ad1357bd2f6",
                    "6e047ed8cb7007034ff15840dd53c92096f0e7ed5befa07808de8afa35d35874",  # safetensors
                    "adbd0baa059074501b7686db2b0c01715f3a317275c2657c5dfbfd6ee92389b7",
                    "eb63790fb32b5660de34fa42c2e608df58f7aa3680b4984f0ee9008fe613729c",
                    "f125c20a33b0ff2dbd4e8ad9acebc34383cb2ef98668169ef79a8c06655ced35",
                    "e64e0ac83a785ef584a0e86b347fae8f9e2bd84324a49396ca8a9fe7532a947b",  # GGUF
                    "70001b3ac1b66522142bb86e4c3e87e20c2bbd07276c763878e0838ef6184aad",
                    "f46fd1e2b5fef3b9f7ae80d183cc77f7be181117a72a0bb933bdef0bc6cd679e",
                    "83676d73726d101325a47c7f8a60cedf10bab99ea79a6bedad7761220cb4a625",
                    "a621a907586e5e270e7c7873b167364d8a935ff347d8240fa9bab319678da690",
                    "f0af1a089f40d8611db5c59469314f1547e2df23c6eff24860359b37ea9bd966",
                    "72478320b8dbfd9aeaea010dcf0896e3116fa5ab940f3b472882d9f9d2d7333f",
                    "9c1a88e36334a48d8482fec54b14ea1d5fd31f0dbb65d13cc616e63dc7c42be5",
                    "d0689f727e8ac4fef3ec4b1f29e8a3bd12e1116559eeefb2a1a457cd4e676d1e",
                    "fea158a4afcfaa6e95e04799bae0287de0c4fcb188f3b41768a46ce48c71c9df",
                    "2e5bc4e73312b5aec4c1a55631cb4ed69cf34ccaa6d1f28f7045f137a579b439",  # cogvideox
                    "015fdecbc3b5369dbcb2302e4b79985437ac4496d1b9ad63316423a222fb0803",  # hyd 1.1
                ],
            },
        ),
        (
            "google/umt5-small",
            "UMT5Model",
            {
                "identifiers": ["encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"],
                "file_256": [
                    "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58",  # wan t2i diffusers
                    "decf9b70814ed5e9965bfca9fbd0483462e2bf743790663025b7742f8c014c72",  # fp16
                    "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6",  # auraflow
                    "c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99",  # wan
                    "7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af",  # wan i2ixxl sai fp16
                    "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",  # wan i2i xxl sai fp8_e4m3fn scaled sai
                    "fa1d36fd54f171ae60fea915c23bd77986b330bbed9729f0d2f8ecbe9168bc48",  # gguf
                    "4a3176f32fd70c0a335b4419fcbf8c86cc875e23498c0fc06f5b4aa0930889e0",
                    "adbc782b9145a27e15d63dfa25057efca0ac75e2db7d372c901ddaa130ca2def",
                    "b7e2ca4c493c9d51fa951005e8ceba2f4b6b6877cfb4c36a8955c6cd68a1dba7",
                    "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc",
                    "9209b4c77b34ad8cf3f06b04c6eaa27e7beeebb348a31f85e3b38a1d719b09ed",
                    "8bc12d80bc0413573fa58a93626117440b4528f640dd9cb310732e05fa9e6c3e",
                    "f64f8d6dc4d8a24276df69d0ccea789aae686f7417950a41e6568c30cb478a5c",
                    "17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40",
                    "eaea358bb438c5d211721a4feecc162000e3636e9cb96f51e216f1f44ebd12ce",
                ],
                "layer_b3": [
                    "cd92b29c9099a640e3f5d4a76e64b3467f87f6c056119e0defdff94d311ad6de",  # wan t2i diff
                    "1c943dbcb8b328a7c6c852921ddaefbd84c9df8c83bc51fe303c1f06cb734102",  # fp16
                    "1639a6467af0db1e15828d33b878e568cba1335947eeadd481170bcdc9ba8e33",
                    "72a0329740dee29a2c099eec3c320b3945590a74293356014c30249fe69652e5",  # wan
                    "0374cba03c607ffe8ab8f04994d82f82e80901dc7578f1a9a6cb2637608be5d5",
                    "d75a407f873e1cfa1a0a36214b53b14bfebe9253ea263465151c07f0d57f3f29",
                    "621153502b985c143d304318c91dc3d10296d24268c81e3538fc336fdc84c915",  # gguf
                    "43bb052945d38a68bec27c3d26162e88e306e6074d027d3b4b2b8ae2b1851691",
                    "98f50ea5d55e61c1478df47e567e48bdd036d240b9129e64d53a826406900adc",
                    "9400313b8eae31699473daa5f840d25a4ef660f68de9a7894f1a28f214f23384",
                    "9f13826b8e4ddde24d80de6a947a7868e26cea25dda52790ee6ed695ff72b9bb",
                    "475773ab108a537ff904b84e7f3a80129ba4983deb7170b6b52c922ece6069ce",
                    "5ef27b3c1eddb08cfe41b452cf9529d86dff811645d40c165bae324486d19e96",
                    "e170559d8551cfe651344594e54c0a9a90c0068b00f3866f6e9a3737e20925cb",
                    "e8dc7442a20bcdc7b6e5dd0265939d88896eab5ddd33ee16f1f09537e65914b8",
                    "4d3d5049857d01741780daf01e96617092973305637b435f4895499a26bbaede",
                    "7a2adadc2372feda23b2169337276adda6d1fdef82ba69f0d3321c4c6ba8c604",
                    "0a7c61a85bb3f51f75924de48ef3f5e87cbf8901f600cbfcae97f5e2919c4148",
                ],
                "layer_256": [
                    "467916d35f3053dce1d40d998fcaf6aa03feda75aa578d964dd61461e23641a3",  # wan i2i diff
                    "58deeef888d4ded4ffababfbf8da27227a4a6ff8adfa42016e12c0180f713816",  # fp16
                    "178ebd3fa3418d33a2e45a80d8b9d3662ff4a8e75f3de3f0332f82c505d8152a",
                    "8700dcb651465fe6c925b7ad6068b58b32951832fff0ed19819510f8d0713ee5",  # wan
                    "954f2129ba166e746c71433f717b572d8869ec14b32b7f214d1701d3b1120047",
                    "32f5fc1daea014b6488b96c2a1330e0aad87e074844fa3e2e3f20b9e58440395",
                    "9245abaf6df8a4b5fcc828ecbcd7b21a1b19bf5f3c4388fb5c8eabc140276dce",
                    "172d0fbbd379ae014a7008e148813818494e9e645db802fd000d443369df9d17",  # gguf
                    "2fa68a26b0386aaf9123d2b4067dafc8631ee724602197dd353f3ea5a61dac8a",
                    "16f0054014e6d07b86b0526d5bcfed7d2aa3aebe3e44e6758933d90cbd3da46e",
                    "fd62047f5d27ff43210c117dc0f253c101e694a5331d6b684688606c92c65ccf",
                    "ddc4f38db9f132fb1b736c1d693b5c039a2d6fe83bdf4f1c1e7a2745b5d79124",
                    "9e9ab11b3ea059b84ae2bcc5be76ab3f730a486d92a16f1fd2a959bdc2ede08f",
                    "bfb178b1ce27f00e122d2328c662fdef6cc239c07efc749aa61ae2d395441b02",
                    "50addf6a911b90194a75b0212429d1af55eb2f9d24715479b9ccc4a40adc299b",
                    "2e46e9f1b714d72160d3b3b775a845b3049a01396fab935f1278d9e8de2ef0c6",
                    "db8d2b49d9042e39d6531b33ec3bebb9cdf42b9e6ad56163f08da2a7da2a53cd",
                    "2d81d19ad5440422b85e0b17c71914269f6c25c9b1fa321c0dd6119ddb41d62d",
                ],
            },
        ),
        (
            "google/gemma2-9b",
            "Gemma2Model",
            {
                "file_256": [
                    "e909230aabafad02d097c7dc02f2ae062b4e6b0593477c1f07679d277e09ce71",  # sana bf16
                    "d61628bc793240439e608c5ae744f55ec8770f684abb63602648a24cb6da60bc",  # lumina 2
                ],
                "layer_b3": [
                    "55a3c812ac0832d154867f5927365bcc776926e48e65f7f35a81fc11f4bb81da",
                    "543572889beb25cad83a43ce70cdd255d2c82951d6595e8c97ff62fd05871c99",
                ],
                "layer_256": [
                    "a0d820c39578cf888f398579d9a00d69b31c81e049795ba70008dad8fe5b3a33",
                    "abc83b04a04467579ea1952a7efbdd252b8641ac0e2a6a9be2a5a73e371111d6",
                ],
            },
        ),
        (
            "google/gemma-7b",
            "GemmaModel",
            {
                "file_256": ["01676b4c6e765f737a5e9854a315de3887e939c370cae116d505777729099a68"],  # lumina next sft d
                "layer_b3": [
                    "438d82c867240f194a4e15798eef2886a911c8f57fa2d9f4ffad1d56e7bd1ccf",
                    "1de38e09f5f2c5345de48b8cd4dddcfff3e341cc0059752446e186b3863f0981",
                ],
                "layer_256": [
                    "e4835a72d582b4ae066d6ff0519f2ee9f8b21fb02e8c28d8eaa317f8d1e9ea75",
                    "1657c7180b48672004f4463308dfdd56d92eedeb23d1408ea766985ca208e5aa",
                ],
            },
        ),
        (
            "google/mt5-small",
            "MT5Model",
            {
                "identifiers": [[250112, 2048], "text_encoders.mt5xl.transformer.shared.weight"],
                "file_256": [
                    "0524484ec81425ba9deef6fac1393a78ba9b1c9bfed704a4be5f9c7255975cc1",  # fp16
                    "32f70f1d187e131a5fc3e4f0edc97ce89360d8e2f1d90177a443a05296097acc",  # fp16 enc
                ],
                "layer_b3": [
                    "a1d616c37711ec7b9073d04734af2f5fd02f9035a322eb46efeace922e104c51",
                    # "bc71d4259f4feaa0fb27c1f288765004840f39247cddc98b3ac37329ff1354d0",  # fp16 enc
                ],
                "layer_256": [
                    "bd337daf0c1aa36896013109b406a0580aa3bb8ab9291d89df3015d737358e95",
                    "2e40c48c96fc7df636aad96d3e78ed0ba9f68c3059e21b7fcf917f284c569a61",  # fp16 enc
                ],
            },
        ),
        (
            "Qwen/Qwen3-15B-A2B",
            "Qwen3MoeModel",
            {
                "file_256": [
                    "c56947057481fb5e7cdf766e442da81717b34addc88bbe8f3728fd25bd03cbae",  # qwen3 coder 53 a35
                ],
                "layer_b3": [
                    "d2d1e0875202f5c9c84c781a2105620250733bd01832f67b2c17bc981d1eb508"  # qwen3 coder 53 a35
                ],
                "layer_256": [
                    "408c01da57c4968b7b0e36d98a74e321153e7aeb058fea63ffd140e323526476",  # qwen3 coder 53 a35
                ],
            },
        ),
        (
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen2VLTextModel",
            {
                "file_256": [
                    "1f48ac458d6fbd0aec53a116065a7ee3f1d34bddde544e25c16a05c9d5392b78",  # orsta 32
                    "0e85c7111ce849293e97aa09ce1172352ecece023a3ecea7ac8311e326b47f3a",  # orsta 7
                    "d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",  # qwen img
                    "e10bd9583a77250376d9134cd6b46799029dfa3b4d7989c1050b3ec149cc7cf5",  # qwen flux
                ],
                "layer_b3": [
                    "e4f681bde70a753f30f83495a2aa340d251bf3d818eb5a1cbe58f85fd6ea0d40",  # orsta 32
                    "47b062ce8ddb14845fb1a71d2fd88fd52a82e26561ba3eb05be057915a867775",  # orsta 7
                    "b6386f70b528ffa9e09fdd8db8a7b91a7c462ed97b06963576c6139e25fdcf31",  # qwen img
                    "4cd449df9f9004a7e53005583a7e4cfa6de42912f03647d2ea799d489e9c1406",  # qwen flux
                ],
                "layer_256": [
                    "ed36a4a11c4ebebb10d1e010cb93e2e43fcaf975cd42bb6c9958537593d0d44d",  # orsta 32
                    "f7f6f64e7b6d7826400a2fc0eef942a47c47bd5914e051ad0c8cd9ff5ff7982b",  # orsta 7
                    "f341ed0f792cf0570ceb21d3b64ed14bf9875e9fcb90116851364eeed683a6ca",  # qwen img
                    "ba031d0da78afe24ae63558ad29b8028244a7bd4750a5615dab9079fe32a5fd7",  # qwen flux
                ],
            },
        ),
        (
            "openai/gpt-oss-120b",
            "GptOssModel",
            {
                "file_256": [
                    "68a8dc1f8e2e5996cb702f14332a25ddf3463daeab2df68e21ca09ef181203c3",  # original model
                    "a881aa5f561b26a22b14a8262aa61849ace349ffd73d74769e030ac90a1fcf8a",  # diffusers
                ],
                "layer_b3": [
                    "b52807536902cabbf84f99e4fa2f8713fb4ef77e739f06367ee0d486e3222faa",  # gguf
                    "43c618018db1fd6e915dead610652da261d9058b73bc5355c85c6ac69af4d913",  # "original model"
                    "ab27ce7391b7fbd6ce3c319faa119afdac68f746af6a0ce2c3400a132f36f6ac",  # diffusers
                ],
                "layer_256": [
                    "de5dcad822be5ed6196f0f3f6965739993118d14db97b33a94a269f4f1b7a363",  # "original model"
                    "575f1977ed42d95a050e13dadaafc05a6d94c8aadca8364dca8a62aa4f2b146c",  # diffusers
                ],
            },
        ),
        (
            "microsoft/Phi-4-multimodal-instruct",
            "Phi4MultimodalModel",
            {
                "file_256": [
                    "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6",  # mini
                ],
                "layer_b3": [
                    "cf4add4ada6082f448788eaf2937f645b5212db88e06ee81475b8be0e99063dc",  # mini
                ],
                "layer_256": [
                    "7ff992b780b2f8993dd6bb9612207943638b2a42badc976ce80893bc205e801b",  # mini
                ],
            },
        ),
        (
            "laion/clap-htsat-fused",
            "ClapModel",
            {
                "file_256": [
                    "c92b5a2bee69ff5dd05820d9e0a5cddbc9c9b9dd19a6cb3214f0cf4f29a4d1b0",  # audio ldm
                    "ae69f555e7f1a2333b8e684c9fa8233f44a47bbadf76d484f941b74f74d2753d",  # music ldm
                ],
                "layer_b3": [
                    "a4d26450ac399d51b9abbe37859615bb02a5cbf63521da4c7cdc549d04a2872c",
                    "ddf310d8eb2d4e3f61e605978675a9d3a748cad9406b9aee8335eae013e77573",  # music ldm
                ],
                "layer_256": [
                    "843ba86000971d6067bfc4f3ed6dd01bd6f6726188aaa15d86b05554f4fe8481",
                    "27529e30442d030a28badf9d62710f4b74e38e9c4424ed169c7e0ac072f5a771",  # musicldm
                ],
            },
        ),
        (
            "google-bert/bert-base-uncased",
            "BertModel",
            {
                "file_256": [
                    "c6c6348af2cb4d5852fe51102ce39605903dbe7925c005cf8995506cc21ea914",  # hunyuandit
                ],
                "layer_b3": [
                    "30d7d2cc3ec9e4ba45844e005d0bbcb5887b6a0976042f73da916237dc5c4c12",
                ],
                "layer_256": [
                    "94fd2508680ff684eff57e4a5a8ca46bf338fc356a9cf6fe8db2b84543dd7971",
                ],
            },
        ),
        (
            "llava-hf/llava-9b",
            "LlavaModel",
            {
                "file_256": [
                    "f5ad57d3eda300a3195bc9c0bb36ab76ebe88831f128e9851e63440aff4a6741",  # hunyuanvideo
                ],
                "layer_b3": [
                    "d7d6ccb9dbba90b64e4cd259b6309e56708b3f4fbd6e9f85e9f0410e549133ef",
                ],
                "layer_256": [
                    "9969c41152aba689413b7f63888ecdc0c0badad2c2960e689ebc4c0e4a696c73",
                ],
            },
        ),
    ]

    additional_tags = [tag_pipe(*entry) for entry in diffusers_addons]
    additional_tags.extend([tag_base_model(*entry) for entry in transformer_details])

    assimilate(
        mir_db,  # format
        additional_tags,
    )


def add_mir_diffusion(mir_db: MIRDatabase):
    """Create MIR entries missing from the database"""

    repo = "microsoft/speecht5_hifigan"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="gan",
            series=series,
            comp=comp,
            file_256=[
                "d9dc6513c30a5b86c2497712690c04fe74b4aa79fdab6d490b34fcb4e24c590c",
            ],
            layer_b3=[
                "85b5acdf29ad04c63f885383340d8e3445ae0055521f82cabb82bd09cfb9a956",
            ],
            layer_256=[
                "bd52b538e7ac05711be9321cfb7619d4056996ce32923c9c91ee02cf69154770",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp="pony-diffusion",
            file_256=["67ab2fd8ec439a89b3fedb15cc65f54336af163c7eb5e4f2acc98f090a29b0b3"],
            layer_b3=["bf4c2154daa4ece7292277b210d081f98759e9ed4d5c889564632e3ccc4a1071"],
            layer_256=["465425d4420dcf5aa4b4d5b456db11a1fcc7c8f61b2e4a87e2470297c98bb96e"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp="pony-diffusion-turbo",
            file_256=[
                "7555ac941f3a767833830ba5cc9a4508a9777cbf97b487b6baf0400ab7000587",  # turbomerge
                "9322f9d91b28abf09e4137bc02ec806af23510221a164e71b81778e61cc3b4b2",  # turbosimple
            ],
            layer_b3=[
                "1e8f23fcd4be0f00eb52368b91c709fffa8a3b8e21772b92b2e0671eed9117d0",
                "5c8b3f34f9d0a58135cf72fbfe9b5d75b5545a10e3d726478543fa7cc510a8bc",
            ],
            layer_256=[
                "7edf51ef09b39c46937a4e4141707c040cd12af0d95299a4d3cd2b7d3fabe035",
                "74e4dbc89d57d61ff7e8af8b0fddcf7466ba233d53ca4ffb7777138991bc3d52",
            ],
        )
    )
    repo = "cagliostrolab/animagine-xl-4.0"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "8ece83aa1bed1fb39a2b81f1660f0ce6889218e493c1f2ed55e9f15f59a7e03f",  # v4
                "6327eca98bfb6538dd7a4edce22484a1bbc57a8cff6b11d075d40da1afb847ac",  # v4 opt
                "1449e5b0b9de87b0f414c5f29cb11ce3b3dc61fa2b320e784c9441720bf7b766",  # v3
                "e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0",  # v3.1
            ],
            layer_b3=[
                "268ffbb120670b9c4b25158bd474c787740884b7738b48203aa03c4c3f00028f",
                "18fda1a55cad137d62c81d4328f5ece85d88b126261e06b9e14ab68055d5d484",
                "bae9bc8a5c43145bcf92ee3391618d9eaddd689f626991bae202de9cf5f1e70e",
                "d6bc5ccafa2b97c867b13a1e7a8c2c7ad9c4877055a66c71bb773557bc306447",
            ],
            layer_256=[
                "c21d1c38813e078817122e12866ab39f5aa7f56945dd4a8beee3cae1e0f139e7",
                "b916c162c981155aaf74e93d5314038af6767bb5a129c51ee05a1fb6a206c6ac",
                "ecc6bfc73824a2d7c3b0ca184854a235859f329c83768f017b07a19a535d17b4",
                "97f6ca05de7fbdae7aacb2427a552f924492176c474a23dd252c192e1c0e9d65",
            ],
        )
    )
    repo = "OnomaAIResearch/Illustrious-XL-v2.0"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "c2a1a3eaa13d4c107dc7e00c3fe830cab427aa026362740ea094745b3422a331",  # v2
                "536863e9f0c13b0ce834e2f8a19ada425ee4f722c0ad3d0051ec7e6adaa8156c",  # 1.1
                "3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9",  # 0.1
                "e3d12d0f76d61aa31d2668a2217e5b642592193f2946842c44d7056ea5469cce",  # 0.1 guided
                "735cf3fefcbdc4f7817f53247e38b836ffd27c7641af6d8daa21d245242cb4bd",  # 1.0
            ],
            layer_b3=[
                "93b061baf21d743d592327a61f027d099d8e18da9808a76c7704ad123eba4a29",
                "dc05fed2acbc73cef4c377cfa2a681c5cf6d065b88d8bf70d371bbcce6a223a8",
                "8eb1c30327e5b71b35b9a4513dc5f2cac9f244667393c0eedb10a26aa9991cd8",
                "3dafbe31f6ebaffa3d054e1b37049e1147faa2474ceb6dab7bc3c4cded0c845e",
                "892533778ee14454938f7b50830093f58e12f1e14560a148f71927e4ccff5f5c",
            ],
            layer_256=[
                "397791b3d77affb7bd35c5ded7377493c6bf456920a41388ba95bd0157109803",
                "b23c02b8519c6777a1f271662f4251a59468c4b3e11184a2d722fa8929b4ea48",
                "a373981494f5508c124a1960bdd096bbc96935fbb54b1218f563206d3892c176",
                "b709df257c40d9d981f686f2880bbe64f43b78805b7213768d659a142a593efd",
                "f1e6b4cab0fce608dca6fa851384e8728202449f16270fbd1f0c4c5ec4946c10",
            ],
        )
    )
    repo = "playgroundai/playground-v2.5-1024px-aesthetic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "11b6d7bce65674659cc6b7ea960658436edfd80e566cb240ebd4bfbc3e2076c8",  # 2.5 diffusers
                "bcaa7dd6780974f000b17b5a6c63e6f867a75c51ffa85c67d6b196882c69b992",  # 2.5 aes sai fp16
                "956dca99114aaa5c3eb526381309d37ee96737e78ed64c8ae613409f47c3f65a",  # 2.5 aes sai
                "933778ce76c1fc0ca918b37e1488411b8a99bbd3279c12f527a3ac995a340864",  # 2.5 fp16 diffusers
                "5c7d38880d0940e6795158b7608ccef89217272b1f2a9331c5b0a2adffcd82c4",  # v2 sai
                "0411e988479884b1a3ecd184123efe38d051d8d0ef24270585a7d1d57499464a",  # v2 sai fp16
            ],
            layer_b3=[
                "d55b22740da2d5b98020ad2390cdc0a7ee08cf9e0d98c11957f16cc20c49815b",  # 2.5 diffusers
                "7e9be9bd9a3aed1ad7207e2f77c98c24c3a75f6adcc9b53514033c6c3365d289",  # 2.5 aes sai fp16
                "5c6dfcc8d01dfb64723f8f5785caa080e2987859c0a050470bfdbe5312be9efc",  # 2.5 aes sai
                "703f775c6e48ed5b0eba6e847414f047bcd4adc677dbc1bf221b3ef05b2ac471",  # 2.5 diffusers fp16
                "72d4ebe4af61f8a7add8fe36b8acd16602894279fb5a744ad50b5b5bac7067b8",  # v2 sai
                "acb757b851db12cdf9d4365a45ee0d6e64afa77ac95583bb82711baf7c4125fd",  # v2 sai fp16
            ],
            layer_256=[
                "adb7be228d4ee6e583c3e5ae4ddb579fef64c3987617ce4d4aff3eb7f8d6a3f7",
                "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843",  # 2.5 aes sai fp16
                "fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20",
                "bc7021473a04a6de3fe0d0fed600875d852ad1ad9d47c445278f66ce9e8ec7a0"  # 2.5 fp16 diffusers
                "fc94481f0c52b21c5ac1fdade8d9c5b210f7239253f86ef21e6198fe393ed60e",  # v2 sai
                "a6f31493ceeb51c88c5239188b9078dc64ba66d3fc5958ad48c119115b06120c",  # v2 sai fp16
            ],
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "precision": "ops.precision.float.F16",
                    "generation": {"num_inference_steps": 50, "guidance_scale": 3},
                }
            },
            identifiers=[
                "edm_mean",
                [1, 4, 1, 1],
                2516,
            ],
        )
    )
    repo = "segmind/Segmind-Vega"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "94762e983e5942056be73c5c1d4464b8ffa1ada500b4fef1267550e2447953ce",  # modelspec sai
                "1ab33e37fbb2566c55cd729e4ab79cc2f99cd9d0a578fabc7a2cf4ee47968be1",  # diffusers
                "8cfa375669b1222d6fecf470f41b2abb370c76a90ab9568964c4bb15b34ec8a2",  # diffusers fp16
            ],
            layer_b3=[
                "2f353c5e6ed0a2c05af00d014e18e65f69f1ce8c48f8eefbf8ad71b34f940fbf",
                "cc34bd3135d7cafc3cb6e3f6e7cb6896c98277bad52877a952ddbd2ffe222e01",
                "b90efdc848f5386d5250b6fb233ce380cf6cc299f497cfa1d2feaef22f87c9d1",
            ],
            layer_256=[
                "029b89ee311110c8f945dbdfc52c1d5daeb1e78c353c38aa3141ec68ce28e7cc",
                "5cdb948e5f3873300679073391d48fc648171f02093d7737d078557ff75762bb",
                "f73afbe43cc76571cb86ebcfced618668a2fb2252b0bc6ba88d6e942bae75741",
            ],
        )
    )
    repo = "segmind/SSD-1B"

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "7cb406ec0662e91570a79f3c4fb8f0ea5325bffe6af5d9382edae838698f72bd",  # modelspec sai
                "1895a00bfc769a00b0c0c43a95e433e79e9db8a85402b45a33e8448785bde94d",  # a1111 aio
                "0bf1ce6b065a6b969ab02dc8e8fa21eb20ee189b10935c49ce68c77a7e432c1c",
                "02ed8ebd0ed55aec686fcf20946d7a1659a31f9f8d9c3798cd254ba6b67434ca",  # diffusers
                "40d8ea9159f3e875278dacc7879442d58c45850cf13c62f5e26681061c51829a",  # diffusers fp16
            ],
            layer_b3=[
                "c074dc38e8ec836816b91cbcc2ca17f80d6106de8d196d416ef9a27c8837ee45",  # modelspec sai
                "1d6c0216da57fe98e7ad29e9653566725f5b2a87845fdbdcda257b3be817b5f4",  # a1111 aio
                "c074dc38e8ec836816b91cbcc2ca17f80d6106de8d196d416ef9a27c8837ee45",
                "89f86d9c846495870416b4945b6a46a517f28405e5bab666feb4057f012340be",
                "535b47e9b70da6494878ca6d45af3f2e201b7f17748432911c12232e586855e6",
            ],
            layer_256=[
                "52267d5d327a2ba92c7a14261a9d081df621b8366819b1bb3a47d130523a813c",
                "b365a3631c6c74532f3a571c84c68e088be35496d35be1e932031713ddd2a2f4",
                "52267d5d327a2ba92c7a14261a9d081df621b8366819b1bb3a47d130523a813c",
                "89f86d9c846495870416b4945b6a46a517f28405e5bab666feb4057f012340be",
                "535b47e9b70da6494878ca6d45af3f2e201b7f17748432911c12232e586855e6",
            ],
        )
    )
    repo = "shuttleai/shuttle-3.1-aesthetic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                2: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "176871da1d5d2d511a52ae9b0dd70faa1f5d1b7734b7e33ed6b4bffa52050e0d",
                "4b80d37681eaed07b7f5b3825a392da929d1620933ede7c2749ef3613cc53f42",
            ],
            layer_b3=[
                "ff422d1734abf33366e87bbf44267dc6096c5d499e695287c35558174877412e",
                "5ad8034eac6b82d842311437101c52b5d35826ce34994940d9e667e702a0d45c",
            ],
            layer_256=[
                "e5d95de314cbfc49b79479118a1ac0b90fc95ccd6bb1a5c95803996d6cebf8fe",
                "d299e8ea4a605917ab98a4a7330d4d398b4ae295efbf458eeeceb5ff1bd7959a",
            ],
        )
    )
    repo = "shuttleai/shuttle-3-diffusion"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                2: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "a5b04df4072698395387c21e8da0176d03f6557e0c38ff1dd3bf469ebab9d0fd",  # fp8
                "a91b46de2055b3511ee87523b57862648856e8c00100161d5b520543a7302755",  # norm
                "23a77c86189d5934da48bf44bb871cf80ba99177ffd3fd5272cdecb208c8b8be",  # mlx q8
                "d3782d5a8f6e82c6676e8e26d54020934ada589d2aceb17fc5ca604b1bd55da8",  # mlx q4
            ],
            layer_b3=[
                "4dd3174edf6b680ce9daf3de643e33ae2c4f09a4d5968da61ea48885f3a193c0",
                "9fdf191b2c58b2a6e190396e12314530593dca4f2a2bee389ec5175da5e52af8",
                "ad203ad6a00d8b1315337e34069e7c41016ea407469a536de8ad6807042017fd",
            ],
            layer_256=[
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
                "7ce8d449b32a9c959431ade729b513ee7a6457f11e1c13e3ef04dd8db3494621",
                "9c3395f67a3d844483b77f0ddd5e2ea64b61732fa9d9da19845bb8ae574c1f8c",
            ],
        )
    )
    repo = "enhanceaiteam/Mystic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={0: {"generation": {"num_inference_steps": 16, "guidance_scale": 7.5, "width": 768, "height": 1024}}},
            file_256=[
                "179d4000e44295f6dfadc0e4ac210146454724d46371b82657200ff9fb5c68a9",  # mlx 0
                "48ca85274e3b67f07f70dd84b67725e62395c2f7b188394342716f783ea4c6ac",  # mlx q8
            ],
            layer_b3=[
                "91074aaebe1b5f3b2e7755d3c092af7eb240e92a192360690f1033949d3c8a68",  # mlx 0
            ],
            layer_256=[
                "3942e6a52dbb0abaf63b031d9c4eda0df47576b51d4c81361978a3dc27b1309e",  # mlx 0
            ],
        )
    )
    repo = "shuttleai/shuttle-jaguar"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                2: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "dcbc4f2470b177eed12c7d7515c0e7342515a849ebd31a50c8d8d43913d7bd32",
                "26a7aa64c0798a3549e1d767932da0a7fb82b49f8edcbdcde804a20d9ed1478f",  # mlx q8
            ],
            layer_b3=[
                "9906c29933d0c33a6ee8d9712f33fa8bd4b35b46a1c7b565ae48832b757dd980",
                "89c453c4bf99220405687eed984dace4492bdae1b6fb08f3d9629145b1a11672",  # mlx q8
            ],
            sha_256=[
                "4eacf27e5659f5dc42f34c407cbe9e1e202290692df754eb68fe913f59fa2941",
            ],
        )
    )
    repo = "freepik/flux.1-lite-8b"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={0: {"generation": {"num_inference_steps": 28}}},
            file_256=[
                "09e970a7b8d1813ea7cacd48f9a944fd223882b137a8f4f3b61d864cdc20bbec",  # mlx q8
                "de90e69945c2f4afcb9b6a057ce48190905c984370fce76b16ba3b97d46e2747",  # mlx q4
            ],
            layer_b3=[
                "9276fa4805efeb45c08cca32c5b51d490e57a2ce5c15ef476a8e468a509c5cdf",
            ],
            layer_256=[
                "e1afe2f9b1ca55b3c659293cf3237f6b5571f5c4e826bad025ff0f7b54dc34ee",
            ],
        )
    )
    repo = "freepik/f-lite-7b"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    repo = "freepik/f-lite-texture"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    repo = "freepik/f-lite"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    repo = "TencentARC/flux-mini"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=["4236455adeaeb4ed444d63b253ec99805022d17e962ed7261ada9c72ce11cfee"],
            layer_b3=["c1a6f83585398fe452d20596a79a522e2986f4c2c01a40e7bfd787af113735d3"],
            layer_256=["e4a0d8cf2034da094518ab058da1d4aea14e00d132c6152a266ec196ffef02d0"],
        ),
    )
    repo = "ostris/Flex.2-preview"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "0407108e446a4f57efffc5e7518bc374876af970d3c6068dc4074de0d221c615",  # modelspec sai
                "df168ba94d5f96c478b24604a6beedff6189047152190509c73c162ea0d8ec02",  # mlx
            ],
            layer_b3=[
                "7f85cdc186896da6965b57d5edb672f08663075d2b207f0e20e328c4034a8076",  # mlx
            ],
            layer_256=[
                "5063de856be5365807d12b47ef6919b4ac611a72651739b2b4050e113bed7a83"  # mlx,
            ],
        ),
    )
    repo = "ostris/Flex.1-alpha"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "5d6dce30a266ccbf530c3a3bf253cd5486720a8fb71cdeed556c28304201dc2f",  # modelspec sai
                "7acf8771b80a91eaa21566abe8c7d9d3ba33d8688e6e98446827749aee7ca1ee",  # mlx
            ],
            layer_b3=[
                "cb3d3edafd81651eefd62894b3572deb02c5304f4b5d4f7ab8654f1fb922ecd6",  # mlx
            ],
            layer_256=[
                "a6b9af6efc25fa77cd24046b81ee66fea09a9987d2a8e56ffca9b7a1c9c9c519"  # mlx,
            ],
        ),
    )
    repo = "tensorart/stable-diffusion-3.5-medium-turbo"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=sd3_series,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                0: {
                    "precision": "ops.precision.bfloat.B16",
                    "generation": {"num_inference_steps": 8, "guidance_scale": 1.5, "height": 1024, "width": 768},
                }
            },
            file_256=[
                "5b0530e8d71b49fa1358f1208047cd789a40bae5b44406c9524b0f0d88f8b246",  # diffusers
                "07119c77c3548a1d9eb30923df4dd55ec74914dc5ec81626804dcbe51ce17a5d",  # sai
                "3c379381344d2a2b3ee3d7a1bc97f7d1e58fa95c6b5187fb48b3ce446f99f17b",  # q4km gguf
                "6b3806cafdb4303ea2638e9e08eb186067b4a46a95ddf344ccdbe56537afaf6e",  # q8km gguf
            ],
            layer_b3=[
                "873821614080a98e1ebfe56673bc96c2ac57379720d4ad2f97e4bca317571d48",  # diffusers
                "7284d2027523482af9ef47405667ca891cc518bfb6ebf1f1d4666cb0accc8cd5",
                "d938ee5738c73f701760ed18acad274b074d2796123aee3f2eee1328b6c36ea4",
                "c4c40056c2a77959083b5a69a1a4b205caa463ccabde057352c5c4e38b2c67b6",
            ],
            layer_256=[
                "3c324055a1ec6eb4ee0242e344bb2b6356afcbd2e215fdd9d160cda691a72fae",
                "7284d2027523482af9ef47405667ca891cc518bfb6ebf1f1d4666cb0accc8cd5",
                "d938ee5738c73f701760ed18acad274b074d2796123aee3f2eee1328b6c36ea4",
                "c4c40056c2a77959083b5a69a1a4b205caa463ccabde057352c5c4e38b2c67b6",
            ],
        ),
    )
    repo = "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=series,
            comp=comp,
            repo=repo,
            file_256=[
                "",
                "",
            ],
            layer_b3=[
                "",
            ],
            layer_256=[""],
        ),
    )
    repo = "OnomaAIResearch/Illustrious-Lumina-v0.03"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=make_mir_tag("Alpha-VLLM/Lumina-Image-2.0")[0],
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "dc6cffcfb0ccfca6332ddb5d2fe25bcb5f496f44b481627f48c42626156fa6a8",  # 2b 22100 ema unified fp32
                "2ac549741fa1c6de2d6cd8be06abcdce52d472eeae2439f948e285258b66a214",  # 0.03 ema
            ],
            layer_b3=[
                "a97b4a63e1e7678e8e7154fae55252267bd1f0ba76b03dba622d801644e657ac",
                "aa6c1b2d1971cea3c4ed0963c8d68d4c50db683f8eab9f77f60ea2d04ed6ce5c",
            ],
            layer_256=[
                "39086c199b9ac296dcba53461ba1e113906d91fbc1b12556d92f5cc77ca11f9f",
                "e51ba2ded40f1af5ca6f78c46eed8305fbd87cd6401e9d439837e10d35cc5828",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp=sdxl_series,
            pkg={
                0: {
                    "hidiffusion": {"apply_hidiffusion": {"timesteps": "StableDiffusionXLTimesteps"}},
                    "generation": {"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5, "num_inference_steps": 10},
                },
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp=sdxl_series,
            pkg={
                0: {
                    "diffusers": "schedulers.scheduling_utils.AysSchedules",
                    "generation": {"timesteps": "StableDiffusionXLTimesteps", "num_inference_steps": 10},
                }
            },
        )
    )
    # possible mixed-type architecture?
    # fusion / united / universal


def add_mir_llm(mir_db: MIRDatabase):
    from nnll.mir.tag import make_mir_tag, tag_base_model

    base_arch, base_series, base_comp = tag_base_model(repo_path="facebook/chameleon-7b", class_name="ChameleonModel")
    repo = "Alpha-VLLM/Lumina-mGPT-7B-1024"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch=base_arch,
            series=base_series,
            comp=series,
            repo=repo,
            pkg={
                0: {
                    "inference_solver": {"FlexARInferenceSolver": {"precision": "bf16", "target_size": 768}},
                    "generation": {"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
                },
                1: {"inference_solver": "ChameleonXLLMXForConditionalGeneration"},
            },
            identifiers=["model.embed_tokens.weight"],
            file_256=[
                "6b71408a7c574d98f00114ab770ac6addc71471770456e482e7b5ec641c02345",
                "1d5d8d5532bae0f32ba35d10d411e506d61e4378dc9fc338f2b1e6af2aa322ec",  # 768
                "a8fe636bbee30fef06dcd8e806ffc65b2aed0ad08a07fdc62f35717d0f851be5",  # 512 multi
                "6420fa13483576d46263996627ba7add2237a01f46dedd3b7750112c0cc2d95b",  # 512
            ],
            layer_b3=["6cd6b3caaea270feb5aff8e9fec205a27da4f48a1e740e63dc9a08f16e70a656"],
            layer_256=["eaa882db6a69cf8ed0104a15b2cdbbb570a23a06ab8c8f65f4c6c21719c6ba25"],
        ),
    )
    repo = "openai/clip-vit-large-patch14"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"transformers": "CLIPTextModel"}},
            identifiers=["text_model.encoder.layers.0.mlp.fc1.weight", "clip-l"],
            file_256=[
                "cb0cba1ead482a850532ebe5ff6b5c8d4456aee32a5228acf0a31e7d9472415e",  # long vit best
                "39e79c916feca4ddf546d9fe923e664714b59ea61074f7228037d17c302f3d17",  # vit l detail improved hit gmp
                "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082",  # flux/shuttle 3 aes
                "778d02eb9e707c3fbaae0b67b79ea0d1399b52e624fb634f2f19375ae7c047c3",  # playground 2.5
                "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",  # playground 2.5 fp16
                "71e183d11db0c6b6282a4d9e0abb74125edc8692393e89ed8ee5571005f35cb1",  # sd3.5 fp16
                "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc",  # sdxl diffusers
                "87c1c0b0894c9e9e10b962e597e8d64dd3a3a2d372c389922b335a53c250b2ae",  # L
                "bd289dd57fee86bc8816b55919a2b03f9c3c75af6025e21777325a6730872325",  # jaguar mlx
                "8377b1ca9d88fe06ec483dd7b3cfc62e5e8dbf8ddd252f455e79d659fa0553c5",  # ssd-1b
                "5487ea0eee9c9a9bff8abd097908d4deff3ae1fa87b3b67397f8b9538139d447",  # ssd-1b fp16
                "92b998a9a64549bfa05c019bde114be6681549a0c79caee903fe30c9444d08b9",  # vega
                "1e090d6a828fd92401be5f83e615fd7b4fb1f4a22e9af9040a38f602e839317c",  # vega fp16
                "11807cb2522cfe99240e5ee2bbeb1ccb42cecca2215102ee872567c7773b28b9",  # flux
                "d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334",  # sd1
                "77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e",  # sd1 fp16
                "b70c11ad5d7e9abf6109348908f599ea382f8019e1f36910bbc8ebecde936633",  # hidream i1
                "fc42badf529dd83f2f7c3d20fe6bda1e22036162f37c4c668b9e130884e20561",
            ],
            layer_b3=[
                "f58a22a381f79985b6d38782f6110a52c2f319b40fdedd3b88b24945dfcbdf64",
                "8faa00b8fd1dbd9286a7237df18caeb8c91af100a6813849b6bae272a01dd7b7",
                "ab5bebc98299c155251a06deccde599ba0128038ee3ce021e8c59a45f58f72c0",
                "c70e9d86a9dcbbbe7c269ef9dfac96ce9c96c46922577338cc1902e5fe936315",
                "f285e9b7b70745df81adc8b558ec74b536b79b6fc02a453ecc61ea9d13f25f1a",
                "7ab17bfa06ab8d65840997ef641f3f593d096860e20141f1eeb0169d131c1c23",
                "2737d3f327e8176dbb549b9c5c4994821430a6c3b07e3bbc925d97511c802636",  # jaguar mlx q8
                "58a826a4a5fe555b4df188a1ebc0d8d9c96cedae3a26ce84c247861dbb93388f",  # sd1
                "1540fd8844898960e18ce8fd153e5f21a8c446bd8c4d6f536a7cf11418f02bf3",  # sd1
                "c4c9caccdbec12b965d93688c521893f75e0bf9a5e0aad70a6a962b669e7b9d5",  # vega
                "e43fae8d5fd1e562607da172369cc0c5ec99b834e42502e682287ff7d12baacc",  # vega fp16
                "c6f79f7416a882891957b815fbdfd6edfaa253c43970b1a25ef14e217599c7bc",  # flux
                "daf5e09f67ad09a909f58a01298fec0132324634cb8fca2a604c3a240c2c453f",  # jaguar mlx q8
                "3f62bfb6bbde05f01435129326166c44aeb113ac0d9f735f31ed3f7dd04f6980",  # hidream i1
                "22f866f3c96a92bc61e9965cf366d706db942ad047ba8cb82109edcd4e68fa40",  # sd3 turbo
                "f3fa9d7a8f15741621c1fe82f8a1bcc5c601c900d947ac09fba7016615a252a5",  # shap-e
            ],
            layer_256=[
                "48daa3d8f939972e69f044533a4312a941971c18c78255f5e555fa26faf664c1",
                "60f5734a74c342be8b0011fc704e718431839790bcfdc7d7004fc39d70f7fec6",
                "6e76e25b4a55dddfa2eecf4b7ab189a8148658a9f6df165c00170f6ce661033c",
                "2d5249df489fec9137cc3a5e9bda499dd9b72a957ddd8e7ad4e99ff3684bad99",
                "3bf085e701713ed3e79775dafea375c3e2a43659ad1ee788b1b393c0aeff9f0e",
                "efb7976800692772e449c81a739339f59394886590ff3f768b0f9ddd87d2a94c",
                "9b0ac8d127c6c457b2eb8c7236f18c4e4ba9e8bbf27130aa8fe854d7c3f7b1e0",
                "24a9ee3d60cdde6c967f08e4b2ec7088fe1bfe308c6896e73caa874860570a5c",
                "5d6d9d0cc7943eb1b8c16862bfd5bee5c3766d0df027ec837e90fac715ac2bd3",
                "68fb122f7d6c3cfbef320341b2af8f5916678e36a69ed36fa8cfcb19e7d5c43d",
                "11807cb2522cfe99240e5ee2bbeb1ccb42cecca2215102ee872567c7773b28b9",
                "50c46cdddbe9f0162278c69b9a1f818519330e3a91b994272e19b5c789670471",  # jaguar mlx q8
                "ffe1c4f55e07c2010ace7b9cf35798bb9f431bc954a32784e5acbdc16acc0364",  # hidream i1
                "146ea48d234e05a934db9d8988e9a9dd86b2ac70f535eaa550ecb0ee23ec135e",  # sd3 turbo
                "d97560cf9704cf71711f6121df2bf55e55a1eda4b574a6ddba074767420bc8c3",
            ],
        )
    )
    repo = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"transformers": "CLIPTextModelWithProjection"}},
            identifiers=["31.self_attn.k_proj.weight", "text_model.encoder.layers.22.mlp.fc1.weight", "clip-g"],
            file_256=[
                "ca18e0c67c1ef1e64cac22926266765b60688f692307ecc06283d987c5768134",  # seaart furry g
                "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",  # modelspec sai
                "fa5b2e6f4c2efc2d82e4b8312faec1a5540eabfc6415126c9a05c8436a530ef4",  # playground 2.5
                "b84f413eebecbd049b72874c1df533a516510cb5a2489ae58c7e320209cf0ebe",  # ssd1b
                "d3df577f6e3799c8e1bd9b40e30133710e02e8e25d0ce48cdcc790e7dfe12d6d",  # ssd1b fp16
                "943a2924ee888295a156dd47089d67181d633b782337890af11ef4b15af17ec5",  # vega
                "5b98e4a57a9292eeb819d67e2d2100f66f17db723cde4ecea27a7c3741160d0c",  # vega fp16
                "4d6effa7a5e600cabf7528ed7234146a13ead1b2c151211d706b293a060b112a",  # hidream i1
            ],
            layer_b3=[
                "d754db276f2d89d2808abb7086b3b8eccee43ac521c128d21a071f3a631474a8",
                "2eb93685b34719e1d1e0541d8902b0a592d95848f80657e32816cf3b152a0f31",
                "e253a5cf3a6242c58037abd6b378bf0281f278e441f28dff7ca1bcfcd3cd6bd8",  # ssd1b
                "16d0eec4e55b0aa63cdca4e4d36f78f66a4b1b9605ce3b1089305026f853c3d2",  # ssd1b fp16
                "f606463295ecf3bae8920d3d45bb9d180793418b3d08c3e84d4c4135c7dc2aa5",  # vega
                "7060993a5eb32d94d1ea8aef7a7301e7be73b199c639c63f8f7cfbfcd2abf10e",  # vega fp16
                "b92af95334c657371af6051a91374a41b5455907fa6622bb66a8c112dc511600",  # hidream i1
            ],
            layer_256=[
                "270e998633eb22145100a3889a62ca270d5080654735e5ff8dda09a7c233af8d",
                "df18800c2a9d9318c4323d991a0fb24a6a9afceb41bea203812f60517c301536",
                "4c228b104f6b9b383e0808c9baa1998957f5125d8f90a4d98c1a86e71edd72dc",  # ssd1b
                "f7fc81d8b5ae91ec28a5106ecc0d067be9a94fd3f394c4aa4686ed131ce5a5b3",  # ssd1b fp16
                "61ab42bd5c0fcb9fd3db1d4014cb844ccae8dc17fd69a108cf077a573d092946",  # vega
                "6c64e36cdda3bec7067e94b05619f882f5d31070792acaadac60ddbef580453a",  # vega fp16
                "43c9e64995b485a7f128771c48defce128640df28e65c7f79537d472f43ebe46",  # hidream i1
            ],
        )
    )
    repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"transformers": "CLIPModel"},
            },
            file_256=[
                "036e6e2bd49697511f4f8b8cb5ee465f93025f7a69a145eadeb9a881ace9b18d",
                "0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5",  # open clip
                "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",  # wan sai
            ],
            layer_b3=[
                "227f26ed63120b9034f4a0c90b6b37eede721a8260f2c1e8f7ea3ccc0d109e7e",
                "3a38ffd1b60499cf2f451f3065079ff26efb9190a86f23ad1c8d993bbeb9af05",  # open clip
                "ce06cf1fd684269ee96631b2bf9334c6ecde6a84a55760dfa0d9d2a6411f28e4",  # wan sai
            ],
            layer_256=[
                "130a94ed12569e099196a6ca27388181922e20148dee5bcb58c5e309acfc2352",
                "cfdbd3fd2b90b64ba12d395a62dd7c3c3ea3e811f0a54593e91bae6516ca5061",  # open clip
                "9125ce5970c649d6f9368c25493d3aaa6b41e224d4cc427e955115f7b7e53d1c",  # wan sai
            ],
        )
    )
    repo = "zai-org/chatglm3-6b"  # formerly THUDM
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="aet",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"transformers": "AutoModel"},
            },
            file_256=[
                "0054d03310248928fdabdeef3fdc753170218dc49a1e9eb5f98323e27683f654",  # kolors
                "b1052386eac358a18add3d0f92521c85ab338979da8eeb08a6499555b857f80d",
            ],
            layer_b3=[
                "a45dfba6a9fa8739777c76deb845fc9589b40f88670d3ce4661646a7b7b1d481",  # kolors
            ],
            layer_256=[
                "174924fd7a07f370bb6fcd1ad07a73eecb7de901f15eefb80f420c1042c47d44",  # kolors
            ],
        )
    )
    base_arch, base_series, base_comp = tag_base_model(repo_path="Qwen/Qwen2-7B-beta", class_name="Qwen2Model")
    repo = "ByteDance-Seed/BAGEL-7B-MoT"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch=base_arch,
            series=base_series,
            comp=series,
            repo=repo,
            pkg={0: {"Bagel": "app"}},
        )
    )


def add_mir_audio(mir_db: MIRDatabase):
    """Create MIR audio modality entries"""
    repo = "facebook/audiogen-medium"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "audiocraft": "models.AudioGen",
                    "generation": {"duration": 5},
                    "stage_2": {
                        "audiocraft": ".data.audioaudio_write",
                        "generation": {"strategy": "loudness", "loudness_compressor": True},
                    },
                }
            },
        )
    )
    repo = "parler-tts/parler-tts-tiny-v1"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "parler_tts": "ParlerTTSForConditionalGeneration",
                    "generation": {"return_tensors": "pt"},
                },
            },
        )
    )
    repo = "Zuellni/snac-24khz-ST"
    series, comp = make_mir_tag(repo)
    (
        mir_db.add(
            mir_entry(
                domain="info",
                arch="gan",
                series=series,
                comp=comp,
                repo=repo,
                pkg={
                    0: {
                        "snac": "SNAC",
                    },
                    "1": {
                        "mlx_audio": "tts.generate.generate_audio",
                    },
                },
                file_256=["e61ae2f638f56ee07a37592cd5a6a9e7d642560ddc78a76ee4a7f96d6922f1be", "973ee1be4032319fd9685ec54eee1b93e79c7bc98c786e67f17c04669714f11d"],
                layer_b3=["18307b00460a64cc4893f9061592ce8d7e15b70fc54065cc8ae0f0155381ec46", "d599b1bb36dee3cee4674b7922fcd69e5ec05b74413f611d21cfdfdf8f9b6119"],
                layer_256=["35ba9aa1feb931010559a178fcac243673d2efdd1396a4b69d406c9853a88300", "5a22c4707ed6c928043f23b59f2d102a579db3a9af41cf6e60d7c3958f182841"],
            )
        ),
    )
    repo = "parler-tts/parler-tts-large-v1"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "parler_tts": "ParlerTTSForConditionalGeneration",
                    "generation": {"return_tensors": "pt"},
                },
            },
        )
    )
    repo = "hexgrad/Kokoro-82M"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="gan",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"kokoro": "KPipeline"},
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
            file_256=[
                "5a5cb3d87478f2e74dfca208ee52209ccfce024095e137097fd276026506e45f",
                "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
            ],
            layer_b3=[
                "3e9b5017cfe67a7804ac717b18b6add42ffc0bd3353490df2bcc520eaaef79b6",
                "379660a87a64524bab69a267e3d9580f04b5eec4f7e3fbd48c6597d164d9b17d",  # safetensors
                "997f154f5a78879ef3ba1a1556977c40b28b9c21076b8f583f752c57ecc36e93"  # pytorch
                "2dc3dba29452b85ea85266084a6248f9e0efe642d5f75b43e64f25b9f2837f92",
            ],
            layer_256=[
                "dbedf0e2115aa309b92689f86534be4a77b91d7900365e1717879fbb19b849f6",
                "2c68574571b3f9229e015a909788116ea2251142e29c1bd5c687863192124e8b",
            ],
        )
    )
    repo = "freddyaboulton/silero-vad"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="stst",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "onnx": "onnx",
                },
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
            file_256=["591f853590d11ddde2f2a54f9e7ccecb2533a8af7716330e8adfa6f3849787a9"],
            layer_b3=[
                "7939427700c3b4d91428a490bde1a6d893f63ee5d79b86f68de9e89c7094d3e7"  # onnx
                "41ca5931452b3ffee588c6c7e5bd327c4e914141604eaf3fd05f4a790ac83bb2",
                "7dc736cd5d840182792bde4edfbf5ddc5aeaf16826a9c72d1ba8166c1e3fab9b",
                "6e2c1bdbad74f56663ffb5710c7cb849a2b91ba331d81acdba47a21f69107434",  # onnx
                "ab5ff443aece9171af5e7603d0b4309d3ecc934e3940ccedefff10f0b54b931e",  # onnx vad
            ],
            layer_256=[
                "2ffef1834d5fe14ad8db58fc78d769d5dc38dda5eddbfc396786f74b326215fd",
                "94ea015f5f7f65b1d8e80f7d52859535e7761d7ed2752e24d57a8d9d9da96672",
            ],
        ),
    )
    repo = "facebook/wav2vec2-conformer-rope-large-960h-ft"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="stst",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "transformers": "Wav2Vec2ConformerForCTC",
                },
            },
            file_256=["97bb9761fb71ec1225100bc81ccf7d002e0d0ba3d0604c1fd2dbda7d7d491f1d"],
            layer_b3=["6c9c5642aa8dce62bcb3eb577bc519619a2d868005c767c5e65371c583a8a8eb"],
            layer_256=["1afcfda68307a75caa1a1c4456cf97e20c7914e8aba828006e9fe17e8675a79d"],
        ),
    )
    repo = "canopylabs/orpheus-3b-0.1-ft"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {
                    "orpheus_tts": "OrpheusModel",
                    "generation": {"max_model_len": 2048},
                },
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
        )
    )
    repo = "OuteAI/OuteTTS-0.3-1B"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series=series,
            comp=comp,
            repo=repo,
            pkg={
                0: {"outetts": "InterfaceHF"},
                1: {
                    "mlx_audio": "tts.generate.generate_audio",
                    "generation": {"audio_format": "wav", "join_audio": True, "verbose": False},
                },
            },
        )
    )


def add_mir_lora(mir_db: MIRDatabase):
    """Create MIR lora entries"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dmd",
            comp=sdxl_series,
            repo="tianweiy/DMD2",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "generation": {"num_inference_steps": 4, "guidance_scale": 0, "timesteps": [999, 749, 499, 249]},
                    "scheduler": {"ops.scheduler.lcm": ""},
                }
            },
            file_256=[
                "b3d9173815a4b595991c3a7a0e0e63ad821080f314a0b2a3cc31ecd7fcf2cbb8",
                "a374289e9446d7f14d2037c4b3770756b7b52c292142a691377c3c755010a1bb",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dpo",
            comp=sdxl_series,
            repo="radames/sdxl-DPO-LoRA",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "generation": {"guidance_scale": 7.5, "num_inference_steps": 4},
                    "scheduler": {"ops.scheduler.dpm": {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "order": 2}},
                },
            },
            file_256=[
                "666f71a833fc41229ec7e8a264fb7b0fcb8bf47a80e366ae7486c18f38ec9fc0",
                "6b1dcbfb234d7b6000948b5b95ccebc8f903450ce2ba1b50bc3456987c9087ad",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp=sdxl_series,
            repo="jasperai/flash-sdxl",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "scheduler": "ops.scheduler.lcm",
                }
            },
            file_256=["afe2ca6e27c4c6087f50ef42772c45d7b0efbc471b76e422492403f9cae724d7"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="pixart-alpha",
            repo="jasperai/flash-pixart",
            pkg={
                0: {"diffusers": {"load_lora_weights": {}}},
            },
            file_256=["99ef037fe3c1fb6d6bbefdbb85ad60df434fcc0577d34c768d752d60cf69681b"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp=sd3_series,
            repo="jasperai/flash-sd3",
            pkg={
                0: {"diffusers": {"load_lora_weights": {}}},
            },
            file_256=["85fce13c36e3739aa42930f745eb9fceb6c53d53fb17e2a687e3234c1a58ee15"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp=sd1_series,
            repo="jasperai/flash-sd",
            pkg={
                0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 4, "guidance_scale": 0}},
            },
            file_256=["99353444c1a0f40719a1b3037049dbd24800317979a73c312025c05af3574a5f"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp=sdxl_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}}},
            file_256={
                "0b97f447b5878323a28fbe7c51ba7acebd21f4d77552ba77b04b11c8911825b6": {"num_inference_steps": 12},
                "55b51334c85061afff5eff7c550b61963c8b8607a5868bbe4f26db49374719b1": {"num_inference_steps": 8},
                "c912df184c5116792d2c604d26c6bc2aa916685f4a793755255cda1c43a3c78a": {"num_inference_steps": 1, "guidance_scale": 0.0},
                "69b25c0187ced301c3603c599c0bc509ac99b8ac34db89a2aecc3d5f77a35187": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "12f81a27d00a751a40d68fd15597091896c5a90f3bd632fb6c475607cbdad76e": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "ca689190e8c46038550384b5675488526cfe5a40d35f82b27acb75c100f417c1": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp=dev_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 0.125}}}},
            file_256={
                "6461f67dfc1a967ae60344c3b3f350877149ccab758c273cc37f5e8a87b5842e": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "e0ab0fdf569cd01a382f19bd87681f628879dea7ad51fe5a3799b6c18c7b2d03": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp=sd3_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 0.125}}}},
            file_256={
                "5b4d0b99d58deb811bdbbe521a06f4dbf56a2e9148ff3211c594e0502b656bc9": {"num_inference_steps": 16},
                "0ee4e529abd17b06d4295e3bb91c0d4ddae393afad86b2b43c4f5eeb9e401602": {"num_inference_steps": 4},
                "fc6a3e73e14ed11e21e4820e960d7befcffe7e333850ada9545f239e9aa6027e": {"num_inference_steps": 8},
            },
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp=sd1_series,
            repo="ByteDance/Hyper-SD",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256={
                "64b98437383537cd968fda6f87a05c33160ece9c79ff4757949a1e212ff78361": {"num_inference_steps": 12},
                "f6123d5b950d5250ab6c33600e27f4dcf71b3099ebf888685e01e9e8117ce482": {"num_inference_steps": 8},
                "a04fd9a535c1e56d38f7590ee72a13fd5ca0409853b4fff021e5a9482cf1ca3b": {"num_inference_steps": 1, "guidance_scale": 0.0},
                "2f26dcc1d883feb07557a552315baae2ca2a04ac08556b08a355a244547e8c3a": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "c5dd058616461ed5053e2b14eec4dbe3fa0eea3b13688642f6d6c80ea2ba5958": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "91fc3186236e956d64dbb4357f2e120c69b968b78af7d2db9884a5ca74d3cd13": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=sdxl_series,
            repo="latent-consistency/lcm-lora-sdxl",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {"fuse": 1.0}},
                    "scheduler": {"ops.scheduler.lcm": {"timestep_spacing": "trailing"}},
                    "generation": {"num_inference_steps": 8},
                },
            },
            file_256=["a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=ssd_series,
            repo="latent-consistency/lcm-lora-ssd-1b",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 8}}},
            file_256=["7adaaa69db6f011058a19fd1d5315fdf19ef79fcd513cdab30e173833fd5c59b"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=vega_series,
            repo="segmind/Segmind-VegaRT",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "gen_kwargs": {"num_inference_steps": 8}}},
            file_256=["9b6e8cd833fa205eaeeed391ca623a6f2546e447470bd1c5dcce3fa8d2f26afb"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp=sd1_series,
            repo="latent-consistency/lcm-lora-sdv1-5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 8}}},
            file_256=["8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lightning",
            comp=sdxl_series,
            repo="ByteDance/SDXL-Lightning",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 4, "guidance_scale": 0}}},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp=sdxl_series,
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256={
                "0365f6107250a4fed1b83e8ae6a070065e026a2ba54bff65f55a50284232bbe6": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "04ea827435d5750e63d113dc509174b4f6e8a069ff8f91970c3d25299c10b1f8": {"num_inference_steps": 16},
                "7eb353b2abcaabab6251ba4e17d6cbe2e763feb0674b0f950555552212b44621": {"num_inference_steps": 16},
                "a85cf70ac16ed42011630a5cd6b5927722cb7c40a2107eff85e2670f9a38c893": {"num_inference_steps": 4},  # float16
                "9f7f13bb019925eacd89aeff678e4fd831f7b60245b986855dff6634aee4eba9": {"num_inference_steps": 4},
                "3b9c970a3e4c0e182931e71b3f769c1956f16c6b06db98b4d67236790d4d0b1d": {"num_inference_steps": 8},
                "7f04ba8911b4c25ef2c7cbf74abcb6daa3b4f0e4bc6a03896bdae7601f2f180b": {"num_inference_steps": 8},
                "13fb038025ce9dad93b8ee1b67fc81bac8affb59a77b67d408d286e0b0365a1d": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "3442eff271aa3b60a094fd6f9169d03e49e4051044a974f6fcf690507959191f": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "242cbe4695fe3f2e248faa71cf53f2ccbf248a316973e4b2f38ab9e34f35a5ab": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "e1f600491bb8e0cd94f41144321e44fdb2cb346447f31e71f6e53f1c24cccfbf": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "d0bf40a7f280829195563486bec7253f043a06b1f218602b20901c367641023e": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "212150d7953627fb89df99aad579d6763645a1cb2ef26b19fee8b398d5e5ff4d": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "e80fcf46d15f4d3821d3d9611bdb3022a4a8b647b2536833b168d317a91e4f74": {"num_inference_steps": 8, "guidance_scale": 0.0},
                "56ed9dc9f51f4bb0d6172e13b7947f215c347fc0da341c8951b2c12b9507d09e": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp=sd1_series,
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256={
                "b80b27dd6504f1c3a7637237dda86bc7e26fa5766da30c4fc853c0a1d46bad31": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "8f605ffde3616592deb37ed8c6bacb83fe98963c1fd0883c2a4f93787098aa45": {"num_inference_steps": 16},
                "fa6acb94f11dba3bf4120af5a12e3c88cd2b9572d43ec1a6fb04eede9f32829e": {"num_inference_steps": 4},
                "bff3d4499718b61455b0757b5f8d98fe23e73a768b538c82ecf91c693b69dbcd": {"num_inference_steps": 8},
                "c7ac2fa3df3a5b7080ebe63f259ab13630014f104c93c3c706d77b05cc48506b": {"num_inference_steps": 16, "guidance_scale": 0.0},
                "4c5f27a727d12146de4b1d987cee3343bca89b085d12b03c45297af05ce88ef4": {"num_inference_steps": 2, "guidance_scale": 0.0},
                "29278bc86274fdfc840961e3c250758ff5e2dc4666d940f103e78630d5b879d3": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "41a7f0b966d18f643d16c4401f0b5ef6b9ef7362c20e17128322f17874709107": {"num_inference_steps": 8, "guidance_scale": 0.0},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp=sd3_series,
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256={
                "8a45878ecc34e53855fe21146cb6ef32682053b7c4eacc013be89fb08c4c19d8": {"num_inference_steps": 2, "guidance_scale": 1.2},
                "9444a5cead551c56c4d1c455ce829ba9f96f01fbcca31294277e0862a6a15b76": {"num_inference_steps": 4, "guidance_scale": 1.2},
                "e365902c208cbc0456ca5e7c41a490f637c15f3f7b98691cbba21f96a8c960b4": {"num_inference_steps": 4, "guidance_scale": 1.2},
                "3550fa018cd0b60d9e36ac94c31b30f27e402d3855ed63e47668bb181b35a0ad": {"num_inference_steps": 4, "guidance_scale": 1.2},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="slam",
            comp=sdxl_series,
            repo="alimama-creative/slam-lora-sdxl",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "scheduler": {"ops.scheduler.lcm": {"timestep_spacing": "trailing"}},
                    "generation": {"num_inference_steps": 4, "guidance_scale": 1},
                }
            },
            file_256=["22569a946b0db645aa3b8eb782c674c8e726a7cc0d655887c21fecf6dfe6ad91"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="slam",
            comp=sd1_series,
            repo="alimama-creative/slam-sd1.5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp=sdxl_series,
            repo="SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"guidance_scale": 5.0}}},
            file_256=["0b9896f30d29daa5eedcfc9e7ad03304df6efc5114508f6ca9c328c0b4f057df"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp=sd1_series,
            repo="SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"guidance_scale": 7.5}}},
            file_256=["1be130c5be2de0beacadd3bf0bafe3bedd7e7a380729932a1e369fb29efa86f4"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp=sdxl_series,
            repo="h1t/TCD-SDXL-LoRA",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {}},
                    "generation": {"num_inference_steps": 4, "guidance_scale": 0, "eta": 0.3},
                    "scheduler": {"ops.scheduler.tcd": {}},
                }
            },
            file_256=["2c777bc60abf41d3eb0fe405d23d73c280a020eea5adf97a82a141592c33feba"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp=sd1_series,
            repo="h1t/TCD-SD15-LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256=["eaecb24a1cda4411eab67275b1d991071216ac93693e8fa0c9226c9df0386232"],
            layer_b3=["90158259812a89beb8874216009c799f420334aac49bbf4fa1bf0ebf4bbd256b"],
            layer_256=["e9825b81bca684126ac3cc8867d2ebc655f74268bc26bea4e4b7e58a52ad6c75"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=sdxl_series,
            file_256=["a599c42a9f4f7494c7f410dbc0fd432cf0242720509e9d52fa41aac7a88d1b69"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=dev_series,
            repo="alimama-creative/FLUX.1-Turbo-Alpha",
            pkg={
                0: {
                    "diffusers": {"load_lora_weights": {"fuse": 0.125}},
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 8, "max_sequence_length": 512},
                }
            },
            file_256=["77f7523a5e9c3da6cfc730c6b07461129fa52997ea06168e9ed5312228aa0bff"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=sd3_series,
            repo="tensorart/stable-diffusion-3.5-medium-turbo",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}, "scheduler": {"ops.scheduler.flow-match": {"shift": 5}}}},
            file_256={"bdcbdfa3ec8ed838b77b1020eea3bc7917a2d42573688a034feb921fde8b1858": {"num_inference_steps": "4"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp=sd3_series,
            repo="tensorart/stable-diffusion-3.5-large-TurboX",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}, "scheduler": {"ops.scheduler.flow-match": {"shift": 5}}}},
            file_256={"fae59d1b749c0d14a8fd4c68cc94eaac92876cee7b91fa75cf8fde3160e09548": {"num_inference_steps": "8"}},
        )
    )


def add_mir_vae(mir_db: MIRDatabase):
    from nnll.mir.tag import make_mir_tag

    """Create MIR VAE missing from the database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=sd3_series,
            repo="madebyollin/taesd3",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["6f79c1397cb9ce1dac363722dbe70147aee0ccca75e28338f8482fe515891399"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=sdxl_series,
            repo="madebyollin/taesdxl",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=sd1_series,
            repo="madebyollin/taesd",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="tae",
            comp=dev_series,
            repo="madebyollin/taef1",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["927f7de7f11bbd3b2d5ce402e608d97a7649e0921a9601995b044e8efc81e449"],
        )
    )
    series, comp = make_mir_tag("Qwen/Qwen-Image")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLQwenImage"},
            },
            file_256=[
                "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",  # diffusers
            ],
            layer_b3=[
                "64af8fb08d2054c81ad2aef94965be8fb1366fcc6136cb9222ae046550af014b",  # diffusers
            ],
            layer_256=[
                "42f255440ef1d379a8a731456bc44312a73a8568716caa6100803990cd5ea7dc",  # diffusers
            ],
        )
    )
    series, comp = make_mir_tag("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    sr_series_t2v, _ = make_mir_tag("Skywork/SkyReels-V2-T2V-14B-720P-Diffusers")
    sr_series_i2v, _ = make_mir_tag("Skywork/SkyReels-V2-I2V-14B-720P-Diffusers")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="wan",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {
                    "diffusers": "AutoencoderKLWan",
                    "precision": "ops.precision.float.F32",
                }
            },
            file_256=[
                "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",  # diffusers
                "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",  # sai
            ],
            layer_b3=[
                "f867543d636029ebfc05b8075e572be0b313a83b0470e56bcf4bbad07a6db010",  # diffusers
                "6b5b229727a2d4e37993687c62c94ff8519a371ab4103c699ff1f5969ca0b433",  # sai
            ],
            layer_256=[
                "121b3974b39263dcca9d644d1b5c9b9251a911b6a8a8e307fcb21ca778e78ed2",
                "364be43a8959012d798d3f98e17d8b5c4b99ba1e70077008dd19acca3ced395e",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="wan",
            comp=sr_series_t2v,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="wan",
            comp=sr_series_i2v,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("Lightricks/LTX-Video")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLLTXVideo"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("rhymes-ai/Allegro")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLAllegro"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("zai-org/CogVideoX-5b-I2V")
    series_fun, _ = make_mir_tag("alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose")
    series_wish, _ = make_mir_tag("BestWishYsh/ConsisID-preview")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="cogvideox",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLCogVideoX"},
            },
            file_256=["a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e"],
            layer_b3=["246addb8dc798240638bffee4546a3c5c83572139b4a2a602d68b4c4146226eb"],
            layer_256=["43c7e9cb4364e55fd563817f01484ede8a09ff19a8e69eb61a32a12f93d6f66e"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="cogvideox",
            comp=series_fun,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="cogvideox",
            comp=series_wish,
            # no repo here, may conflict
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("nvidia/Cosmos-1.0-Diffusion-7B-Video2World")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLCosmos"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLMagvit"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("hunyuanvideo-community/HunyuanVideo-I2V")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLHunyuanVideo"},
            },
            file_256=[
                "95d1fc707c1421ccd88ea542838ab4c5d45a5babb48205bac9ce0985525f9818",  # pt,
                "7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883",
                "fbe5ea338431bc8ba20f7019b474e83379fe5763abfd562adcc04b1c0d35c728",
                "019973c147e0c3462629d8d06bdbdbb83408f3ebd4ea4b4ae21a99c3cdcb54c0",
            ],
            # layer_b3=[],
            # layer_256=[],
        )
    )
    series, comp = make_mir_tag("genmo/mochi-1-preview")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLMochi"},
            },
            file_256=[],
            layer_b3=[],
            layer_256=[],
        )
    )
    series, comp = make_mir_tag("rhymes-ai/Allegro")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {
                    "diffusers": "AutoencoderKLAllegro",
                },
            },
            file_256=["47871a698b18f92f15019d361a81cbc8af4676f8eef9a47fd2b95354a39f831a"],
            layer_b3=["93654cbab7541504d2377c66e72943c7fd9947fca2eb1be01bcc8877c322c1e0"],
            layer_256=["bfd496586118165a13243997101fc7cdd4f855b2d8a73ee2b771a4484c4c2f9f"],
        )
    )
    series, comp = make_mir_tag("cvssp/audioldm-s-full-v2")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {
                    "diffusers": "AutoencoderKL",
                },
            },
            file_256=["42f64f7565b23eabde68c9694e39f18b8bba5f7a14f477e7ed4b51e0ea7de8a5"],
            layer_b3=["00959677dae940b9cfdbe5380c8cbb5a6b4951864cd26f8211d74a3d22b4f3de"],
            layer_256=["54d075953d5253a3abac651de070736c1d5510b857a8ab24c624304f428146b6"],
        )
    )

    series, comp = make_mir_tag("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="dc",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderDC"},
            },
            file_256=["15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f"],
            layer_b3=["cf4ecc6697d18b0663e4eac58203f1dd6d9fb689cf99adfeadbc0019de0c73d0"],
            layer_256=["abfc39d1a6d71f03dde7bc40fec4a90478a97d17ae1688be9aad00e0512b9bde"],
        )
    )
    series, comp = make_mir_tag("stabilityai/stable-audio-open-1.0")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="oobleck",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderOobleck"},
            },
            # file_256=[],
            # layer_b3=[],
            # layer_256=[],
        )
    )
    series, comp = make_mir_tag("stable-video-diffusion-img2vid-xt")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKLTemporalDecoder"},
            },
            # file_256=[],
            # layer_b3=[],
            # layer_256=[],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=sdxl_series,
            repo="madebyollin/sdxl-vae-fp16-fix",
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1",  # modelspec sai
                "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c",  # diffusers
                "78f6189c8492013e3cac81637a1f657f790a237387f8a9dfd6bfa5fee28eb646",  # ssd1b diffusers
                "6353737672c94b96174cb590f711eac6edf2fcce5b6e91aa9d73c5adc589ee48",  # ssd1b diffusers fp16
                "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68",  # kolors fp16
                "1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341",  # vega / lumina next sft d / auraflow
                "703abdcd7c389316b5128faa9b750a530ea1680b453170b27afebac5e4db30c4",  # pixart a
                "98a14dc6fe8d71c83576f135a87c61a16561c9c080abba418d2cc976ee034f88",  # hyd 1.1
            ],
            layer_b3=[
                "bd5b356b509814025a9cf692710b87116d4fcd0e30a8232ed1db133e908d0e74",  # modelspec sai
                "9106380403dee83238af63ff1738396d2fdff9f6d78d0d9c1d0bf770ae4294d0",  # diffusers
                # "245070a60a25ca080cb4951220c3fb1503da43829930d5f6f7a6770b491eafe1",
                # "50e65a628b5fe379798d8956e4a4e1d4b105c84b329f088d577f7f28c22abc49",  # diffusers fp16 matches sd1
            ],
            layer_256=[
                "c9399a4cd39a180a0bb2af96a8297b9330541e090c21e83317cebb2f7cc651da",  # modelspec sai
                "2240ae134a3b983abf45200c198f07e3d8068012fbbd2f658bbaa1fd6a0629c0",  # diffusers
                # "35641f65ad7ea600cb931dcab556f7503279f1d8d99eda170fe7976d48502a2a",  # diffusers fp16 matches sd1
            ],
        )
    )
    repo = "shuttleai/shuttle-jaguar"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=make_mir_tag(repo)[0],
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "6fdfa2add4f04d94f36157cbb0197f97966b612e3f8eff4095315aefea74b904",
            ],  # q8,
            layer_b3=[
                "0ebf9b7010accc44e219e355dd24bf1e3128004093c0c1dfc06f88c0a39fdbdd",
                "d0e7ef3c4af06fa08b4c0485a073e2df55f7b1e9e3ba8f7b261688bc562568f0",  # q8
            ],
            layer_256=[
                "9b28f36873ea283905094a64e1ccb7cfc2b0f0aa166201d0ca63807ac37caa7b",  # q8
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=dev_series,
            # no repo here, may conflict
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",  # dev
                "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",  # dev diffusers
                "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94",  # lumina2
                "8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc",  # sd3 turbo
            ],
            layer_b3=[
                # "245070a60a25ca080cb4951220c3fb1503da43829930d5f6f7a6770b491eafe1",
                "b6db93ed78c4a10d69e80831c1b8fbc1447f04e9b3d494889ee2056b98d41f17",  # diffusers
                "a8a3ebdec4d7b38d65b7169d3604c19b587330e5e66f69ebf0ded56a24ec6903",  # lumina2
            ],
            layer_256=[
                "7950e4f3897c75affaa5f9f3c51c88b4d9a27bfd9b05ad41c3f71d8c1c620b89",
                "79d2bfe93a2ac037cdc59ccb5576e32d00d75d4741fba49fc7e82b9724928216",  # diffusers
                "8f084dc91fd5b481875bc9c86a4ef05e5f176896b7d31c6a5c2ce45c2e174004",  # dev diffusers
                "322e01bd511e20bc2a3c27cd611f81ed85f0046b7c023b5622c2c9a5b8b34f80",  # lumina2
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="eq",
            comp=sdxl_series,
            repo="KBlueLeaf/EQ-SDXL-VAE",
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="ms-lc-eq",
            comp=sdxl_series,
            repo="Anzhc/MS-LC-EQ-D-VR_VAE",
            pkg={
                0: {
                    "diffusers": "AutoencoderKL",
                },
            },
        )
    )
    repo = "ucsd-reach/musicldm"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=make_mir_tag(repo)[0],
            # no repo here, may conflict
            file_256=[
                "16e0c6c7c34e459c19500cc15cf538e6331db14969ea15917caa9b0966e44fd4",
            ],  # q8,
            layer_b3=[
                "c5c32b3fb3e73799838836ccce27d883254254daecd10f86ba8ddc55214014e0",
            ],
            layer_256=[
                "1610c0ce39d1379091eb9ab2a4d14a8567e0f1a5dc6cca40fc0fa6f8e4e97c0f",
            ],
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="kl",
            comp=sd1_series,
            pkg={
                0: {"diffusers": "AutoencoderKL"},
            },
            file_256=[
                "0b204ad0cae549e0a7e298d803d57e36363760dec71c63109c1da3e1147ec520",  # ckpt ema original ema pruned
                "95f26a5ab04779d5467d1fcecaf93160ffa523afe399b835b3e1bb77ff2d937a",  # safetensors ema original ema pruned
                "32db726da04f06c1b6b14c0043ce115cc87a501482945c5add89a40d838fcb46",  # safetensors ema diffusers
                "c6a580b13a5bc05a5e16e4dbb80608ff2ec251a162311590c1f34c013d7f3dab",  # ckpt mse original ema pruned
                "735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75",  # safetensors mse original ema pruned
                "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",  # safetensors mse diffusers
                "a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c",  # safetensors diffusers
                "4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c",  # safetensors fp16 diffusers
            ],
            layer_b3=[
                "82e2dc440a23d78bb91df8c9fce069a8512da51f8f54ea29e3431f545808171e",  # safetensors original
                "2230487833925a104bee96e7ecfebaa4c3c43cc426c7a5b863f2584313dd4833",  # safetensors diffusers
            ],
            layer_256=[
                "e43f3a227b5ecb43a6272fa92ed6011d2e9abcadadd1032dfa7ea7f875f9d5bd",  # safetensors original
                "2494154245becf98891be884f943276aa3f54e9b3f0ea1042903fc15fba488f3",  # safetensors diffusers
            ],
        )
    )
