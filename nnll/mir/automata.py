# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""自動化索引"""

from logging import INFO, Logger
from typing import Dict, List, Tuple

from nnll.mir.maid import MIRDatabase
from nnll.mir.mir import mir_entry

nfo_obj = Logger(INFO)
nfo = nfo_obj.info


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
#                     "precision": "ops.precision.bfloat.b16",
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
    """
    Merge new data into a pre-generated MIR database, updating while preserving existing data structures.\n
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

    for arch, series, new_data in data_tuple:
        mir_data = mir_db.database[f"{arch}.{series}"]

        for comp, field_data in new_data.items():
            if not isinstance(field_data, dict):
                raise TypeError("Test")

            # dbuq(f"{arch}.{series} : {comp}")
            update_nested_dict(mir_data.setdefault(comp, {}), field_data)

            if series == "stable-diffusion-xl":
                for field, field_data in field_data.items():
                    if isinstance(field_data, dict):
                        for definition, sub_def_data in field_data.items():
                            # dbug(definition)
                            if isinstance(sub_def_data, dict):
                                mir_data[comp][field].setdefault(definition, {})
                                update_nested_dict(mir_data[comp][field][definition], sub_def_data)


def auto_hub(mir_db: MIRDatabase):
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


def auto_dtype(mir_db: MIRDatabase):
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
        variant_name = class_name.replace("bfloat", "bf")
        variant_name = class_name.replace("float", "fp")
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


def auto_schedulers(mir_db: MIRDatabase):
    """Create mir info database"""
    import re

    try:
        from diffusers import _import_structure

        for series_name in _import_structure["schedulers"]:
            if series_name != "SchedulerMixin":
                class_name = series_name
                patterns = [r"Multistep", r"Solver", r"Discrete", r"Scheduler"]
                for scheduler in patterns:
                    compiled = re.compile(scheduler)
                    match = re.search(compiled, series_name)
                    if match:
                        comp_name = match.group()
                        break
                for pattern in patterns:
                    series_name = re.sub(pattern, "", series_name)
                series_name.lower()
                mir_db.add(
                    mir_entry(
                        domain="ops",
                        arch="scheduler",
                        series=series_name,
                        comp=comp_name.lower(),
                        pkg={0: {"diffusers": class_name}},
                    )
                )
    except (ImportError, ModuleNotFoundError):  # as error_log:
        pass  # dbug(error_log)


# def auto_gan etc etc
# ai-forever/Real-ESRGAN


def auto_detail(mir_db: MIRDatabase):
    """Create mir unet info database"""

    data_tuple = [
        (
            "info.unet",
            "stable-diffusion-xl",
            {
                "base": {
                    "pkg": {
                        0: {
                            "generation": {
                                "denoising_end": 0.8,
                                "output_type": "latent",
                                "safety_checker": False,
                                "width": 1024,
                                "height": 1024,
                            },
                        },
                        1: {"diffusers": "DiffusionPipeline"},
                    },
                    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
                    "identifiers": ["logit_scale", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight", "add_embedding.linear_2.bias"],
                },
            },
        ),
        (
            "info.dit",
            "auraflow",
            {"*": {"identifiers": [[8192, 3072], "mlpX.c_fc2.weight", "joint_transformer_blocks.2.ff_context.linear_2.weight"]}},
        ),
        (
            "info.dit",
            "hunyuandit-v1-diffusers",
            {"*": {"identifiers": ["extra_embedder", "model.blocks", "skip_norm.weight"]}},
        ),
        (
            "info.dit",
            "lumina-next-sft-diffusers",
            {"*": {"identifiers": ["time_caption", "feed_forward"]}},
        ),
        (
            "info.dit",
            "pixart-sigma-xl-2-ms",
            {"*": {"identifiers": ["adaln_single", "scale_shift_table"]}},
        ),
        (
            "info.dit",
            "pixart-xl-2-ms",
            {"*": {"identifiers": ["aspect_ratio", "y_embedding", "emb.resolution", "caption_projection"]}},
        ),
        (
            "info.art",
            "lumina-mgpt",
            {"*": {"identifiers": ["model.embed_tokens.weight"]}},
        ),
        (
            "info.dit",
            "stable-diffusion-3",
            {"*": {"identifiers": ["model.diffusion_model.joint_blocks.", "transformer_blocks.21.norm1_context.linear.weight", "transformer_blocks.31.norm1_context.linear.weight", "blocks.11.ff.net.2.weight"]}},
        ),
        (
            "info.unet",
            "stable-diffusion-v1",
            {"*": {"identifiers": ["up_blocks.3.attentions.0.transformer_blocks.0.norm3.weight"]}},
        ),
        (
            "info.stst",
            "t5",
            {"*": {"identifiers": ["encoder.block.0.layer.1.DenseReluDense.wi.weight"]}},
        ),
        (
            "info.stst",
            "umt5",
            {"*": {"identifiers": ["encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"]}},
        ),
        (
            "info.stst",
            "mt5",
            {"*": {"identifiers": [[250112, 2048], "text_encoders.mt5xl.transformer.shared.weight"]}},
        ),
        (
            "info.unet",
            "kolors-diffusers",
            {
                "diffusers": {
                    "pkg": {
                        0: {
                            "precision": "ops.precision.float.f16",
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
                    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
                    "identifiers": [".DenseReluDense.wi.weight", "encoder_hid_proj.weight"],
                }
            },
        ),
        (
            "info.unet",
            "stable-cascade",
            {
                "prior": {
                    "pkg": {
                        0: {
                            "precision": "ops.precision.bfloat.b16",
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
                    "layer_256": [
                        "2b6986954d9d2b0c702911504f78f5021843bd7050bb10444d70fa915cb495ea",
                        "2aa5a461c4cd0e2079e81554081854a2fa01f9b876d7124c8fff9bf1308b9df7",
                        "ce474fd5da12f1d465a9d236d61ea7e98458c1b9d58d35bb8412b2acb9594f08",
                        "1b035ba92da6bec0a9542219d12376c0164f214f222955024c884e1ab08ec611",
                        "22a49dc9d213d5caf712fbf755f30328bc2f4cbdc322bcef26dfcee82f02f147",
                    ],
                    "identifiers": ["down_blocks.0.2.kv_mapper", "previewer", "backbone"],
                },
            },
        ),
        (
            "info.dit",
            "flux-1-dev",
            {
                "base": {
                    "pkg": {
                        0: {
                            "precision": "ops.precision.bfloat.b16",
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
                    "layer_256": [
                        "ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881",
                        "20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9",
                    ],
                    "identifiers": [
                        "double_blocks.12.txt_mod.lin.weight",
                        "add_q_proj.weight",
                        "single_transformer_blocks.9.norm.linear.weight",
                    ],
                }
            },
        ),
        (
            "info.dit",
            "flux-1-schnell",
            {
                "base": {
                    "pkg": {
                        0: {
                            "precision": "ops.precision.bfloat.b16",
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
                }
            },
        ),
        (
            "info.unet",
            "stable-cascade",
            {
                "base": {
                    "pkg": {  # prior=prior_unet
                        0: {
                            "generation": {  # image_embeddings=prior_output.image_embeddings,
                                "negative_prompt": "",
                                "guidance_scale": 0.0,
                                "output_type": "pil",
                                "num_inference_steps": 10,
                            },
                            "precision": "ops.precision.bfloat.b16",
                        },
                        "layer_256": [
                            "fde5a91a908e8cb969f97bcd20e852fb028cc039a19633b0e1559ae41edeb16f",
                            "24fa8b55d12bf904878b7f2cda47c04c1a92da702fe149e28341686c080dfd4f",
                            "a7c96afb54e60386b7d077bf3f00d04596f4b877d58e6a577f0e1a08dc4a0190",
                            "f1300b9ffe051640555bfeee245813e440076ef90b669332a7f9fb35fffb93e8",
                            "047fa405c9cd5ad054d8f8c8baa2294fbc663e4121828b22cb190f7057842a64",
                        ],
                        "identifiers": ["0.2.channelwise", "clip_mapper.bias", ".12.self_attn.k_proj.weight"],
                    }
                }
            },
        ),
        ("info.aet", "wavlm", {"kokoro": {"repo": "hexgrad/Kokoro-82M"}}),
    ]
    assimilate(mir_db, data_tuple)


def auto_supplement(mir_db: MIRDatabase):
    """Create MIR entries missing from the database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="refiner",
            repo="stabilityai/stable-diffusion-xl-refiner-1.0",
            layer_256=["8c2d0d32cff5a74786480bbaa932ee504bb140f97efdd1a3815f14a610cf6e4a"],
            identifiers=["r'conditioner.embedders.0.model.transformer.resblocks.d+.mlp.c_proj.bias'"],
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"num_inference_steps": 40, "denoising_end": 0.8},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="pony-diffusion",
            layer_256=["d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="animagine",
            layer_256=["31164c11db41b007f15c94651a8b1fa4d24097c18782d20fabe13c59ee07aa3a"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="illustrious",
            layer_256=["c4a8d365e7fe07c6dbdd52be922aa6dc23215142342e3e7f8f967f1a123a6982"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="turbo",
            layer_256=["fc94481f0c52b21c5ac1fdade8d9c5b210f7239253f86ef21e6198fe393ed60e"],
            file_256=["a599c42a9f4f7494c7f410dbc0fd432cf0242720509e9d52fa41aac7a88d1b69"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2-5-base",
            layer_256=["a6f31493ceeb51c88c5239188b9078dc64ba66d3fc5958ad48c119115b06120c"],
            identifiers=["edm_mean", [1, 4, 1, 1], 2516],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2-5-aesthetic",
            repo="playgroundai/playground-v2.5-1024px-aesthetic",
            layer_256=[
                "fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20",
                "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843",
            ],
            pkg={
                0: {
                    "precision": "ops.precision.float.f16",
                    "generation": {"num_inference_steps": 50, "guidance_scale": 3},
                }
            },
            identifiers=[
                [1, 4, 1, 1],
                2516,
                "edm_mean",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-schnell",
            comp="shuttle-3.1-aesthetic",
            repo="shuttleai/shuttle-3.1-aesthetic",
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-schnell",
            comp="shuttle-3-diffusion",
            repo="shuttleai/shuttle-3-diffusion",
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-schnell",
            comp="shuttle-jaguar",
            repo="shuttleai/shuttle-jaguar",
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-dev",
            comp="f-lite-8b",
            repo="freepik/flux.1-lite-8b",
            pkg={0: {"generation": {"num_inference_steps": 28}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-dev",
            comp="f-lite-7b",
            repo="freepik/f-lite-7b",
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-dev",
            comp="f-lite-texture",
            repo="freepik/f-lite-texture",
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-dev",
            comp="f-lite",
            repo="freepik/f-lite",
            pkg={0: {"f_lite": "FLitePipeline", "generation": {"num_inference_steps": 28}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-dev",
            comp="hybrid",
            layer_256=[
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1-dev",
            comp="mini",
            repo="TencentARC/flux-mini",
            layer_256=["e4a0d8cf2034da094518ab058da1d4aea14e00d132c6152a266ec196ffef02d0"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="audiogen",
            comp="medium-1-5b",
            repo="facebook/audiogen-medium",
            pkg={
                0: {
                    "audiocraft.models": {"AudioGen": {"duration": 5}},
                    "audiocraft.data.audio": {"audio_write": {"strategy": "loudness", "loudness_compressor": True}},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="parler-tts",
            comp="tiny-v1",
            repo="parler-tts/parler-tts-tiny-v1",
            pkg={
                0: {
                    "parler_tts": "ParlerTTSForConditionalGeneration",
                    "transformers": {"AutoTokenizer": {"return_tensors": "pt"}},
                },
                # 1: {"mlx_audio": {"tts.generate.generate_audio": {"audio_format": "wav", "join_audio": True, "verbose": False}}},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="parler-tts",
            comp="large-v1",
            repo="parler-tts/parler-tts-large-v1",
            pkg={
                0: {
                    "parler_tts": "ParlerTTSForConditionalGeneration",
                    "transformers": {"AutoTokenizer": {"return_tensors": "pt"}},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="gan",
            series="kokoro",
            comp="82m",
            repo="hexgrad/Kokoro-82M",
            pkg={
                0: {"kokoro": "KPipeline"},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="gan",
            series="kokoro",
            comp="82m",
            repo="hexgrad/Kokoro-82M",
            pkg={
                0: {"mlx_audio ": "tts.generate.generate_audio"},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="lumina-mgpt",
            comp="7B-768",
            repo="Alpha-VLLM/Lumina-mGPT-7B-768",
            pkg={
                0: {
                    "inference_solver": {"FlexARInferenceSolver": {"precision": "bf16", "target_size": 768}},
                    "generation": {"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
                }
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="orpheus",
            comp="3b-0-1-ft",
            repo="canopylabs/orpheus-3b-0.1-ft",
            pkg={
                0: {"orpheus_tts": {"OrpheusModel": {"max_model_len": 2048}}},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="orpheus",
            comp="3b-0-1-ft-4b",
            repo="mlx-community/orpheus-3b-0.1-ft-4bit",
            pkg={
                0: {"mlx_audio": {"tts.generate.generate_audio": {"audio_format": "wav", "join_audio": True, "verbose": False}}},
            },
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="outetts-0-3",
            comp="1b",
            repo="OuteAI/OuteTTS-0.3-1B",
            pkg={
                0: {"outetts": "InterfaceHF"},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="OuteTTS-1.0-0",
            comp="6B-4bit",
            repo="mlx-community/OuteTTS-1.0-0.6B-4bit",
            pkg={
                0: {"mlx_audio": {"tts.generate.generate_audio": {"audio_format": "wav", "join_audio": True, "verbose": False}}},
            },
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="stst",
            series="t5",
            comp="xxl",
            repo="google/t5-v1_1-xxl",
            pkg={0: {"diffusers": "T5ForConditionalGeneration"}},
            identifiers=[[4096], "encoder.embed_tokens.weight", "text_encoders.t5xxl.transformer.shared.weight", "t5xxl"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series="openclip",
            comp="vit-l-14",
            repo="openai/clip-vit-large-patch14",
            pkg={0: {"diffusers": "CLIPModel"}},
            identifiers=["text_model.encoder.layers.0.mlp.fc1.weight", "clip-l"],
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series="openclip",
            comp="vit-g-14",
            repo="laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
            pkg={0: {"diffusers": "CLIPModelwithProjection"}},
            identifiers=["31.self_attn.k_proj.weight", "text_model.encoder.layers.22.mlp.fc1.weight", "clip-g"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="aet",
            series="bagel",
            comp="7B-MoT",
            repo="ByteDance-Seed/BAGEL-7B-MoT",
            pkg={0: {"Bagel": "app"}},
        )
    )

    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-xl",
            pkg={
                0: {
                    "diffusers.schedulers.scheduling_utils": {
                        "AysSchedules": {"timesteps": "StableDiffusionXLTimesteps", "num_inference_steps": 10},
                    }
                }
            },
        )
    )
    # possible mixed-type architecture?
    # fusion / united / universal


def auto_lora(mir_db: MIRDatabase):
    """Create MIR lora entries"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dmd",
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-3",
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
            comp="stable-diffusion-v1",
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
            comp="stable-diffusion-xl",
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
            comp="flux-1-dev",
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
            comp="stable-diffusion-3",
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
            comp="stable-diffusion-v1",
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
            comp="stable-diffusion-xl",
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
            comp="ssd-1b",
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
            comp="vega",
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
            comp="stable-diffusion-v1",
            repo="latent-consistency/lcm-lora-sdv1-5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 8}}},
            file_256=["8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4", "eaecb24a1cda4411eab67275b1d991071216ac93693e8fa0c9226c9df0386232"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lightning",
            comp="stable-diffusion-xl",
            repo="ByteDance/SDXL-Lightning",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 4, "guidance_scale": 0}}},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp="stable-diffusion-xl",
            repo="wangfuyun/PCM_Weights",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256={
                "0365f6107250a4fed1b83e8ae6a070065e026a2ba54bff65f55a50284232bbe6": {"num_inference_steps": 4, "guidance_scale": 0.0},
                "04ea827435d5750e63d113dc509174b4f6e8a069ff8f91970c3d25299c10b1f8": {"num_inference_steps": 16},
                "7eb353b2abcaabab6251ba4e17d6cbe2e763feb0674b0f950555552212b44621": {"num_inference_steps": 16},
                "a85cf70ac16ed42011630a5cd6b5927722cb7c40a2107eff85e2670f9a38c893": {"num_inference_steps": 4},
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
            comp="stable-diffusion-v1",
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
            comp="stable-diffusion-3",
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
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-v1",
            repo="alimama-creative/slam-sd1.5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-v1",
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
            comp="stable-diffusion-xl",
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
            comp="stable-diffusion-v1",
            repo="h1t/TCD-SD15-LoRA",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
            file_256=["eaecb24a1cda4411eab67275b1d991071216ac93693e8fa0c9226c9df0386232"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp="flux-1-dev",
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
            comp="stable-diffusion-3",
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
            comp="stable-diffusion-3",
            repo="tensorart/stable-diffusion-3.5-large-TurboX",
            pkg={0: {"diffusers": {"load_lora_weights": {"fuse": 1.0}}, "scheduler": {"ops.scheduler.flow-match": {"shift": 5}}}},
            file_256={"fae59d1b749c0d14a8fd4c68cc94eaac92876cee7b91fa75cf8fde3160e09548": {"num_inference_steps": "8"}},
        )
    )


def auto_vae(mir_db: MIRDatabase):
    """Create MIR VAE missing from the dtabase"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="taesd",
            comp="stable-diffusion-3",
            repo="madebyollin/taesd3",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["6f79c1397cb9ce1dac363722dbe70147aee0ccca75e28338f8482fe515891399"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="taesd",
            comp="stable-diffusion-xl",
            repo="madebyollin/taesdxl",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="taesd",
            comp="stable-diffusion-v1",
            repo="madebyollin/taesd",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vae",
            series="taesd",
            comp="flux-1",
            repo="madebyollin/taef1 ",
            pkg={0: {"diffusers": "AutoencoderTiny"}},
            file_256=["927f7de7f11bbd3b2d5ce402e608d97a7649e0921a9601995b044e8efc81e449"],
        )
    )
