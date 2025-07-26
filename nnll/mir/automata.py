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
            print(field_data)
            if not isinstance(field_data, dict):
                raise TypeError(f"{field_data} <-- Cannot combine with database: Not `dict()`")

            # dbuq(f"{arch}.{series} : {comp}")
            update_nested_dict(mir_data.setdefault(comp, {}), field_data)

            if series == "stable-diffusion-xl-1":
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
            "stable-diffusion-xl-1",
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
                    "layer_b3": ["8be44fa13c1efa60f8bcadaa57f1d718473f9660f03c4f0e65dc037960d8cba1"],
                    "identifiers": ["logit_scale", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight", "add_embedding.linear_2.bias"],
                },
            },
        ),
        (
            "info.dit",
            "chroma",
            {
                "*": {
                    "pkg": {
                        1: {
                            "generation": {"neg_text": "", "num_steps": "28", "latent_size": [64, 64]},
                        }
                    },
                    "file_256": ["2c41e8a9831f3be1eaff2c2ed590abb62e4534e814f7ec58a5fd74ff71dc2036"],
                    "layer_b3": ["15e227ced8a89c41abaa9cc44f84dfffdf5ead0c626035e5a2dde2bbb0935479"],
                    "layer_256": ["a4daa6ff6f45ca70c738adb8c19bc3b6f228df931e6bf2a3394463e4dd7ec882"],
                },
            },
        ),
        (
            "info.dit",
            "auraflow",
            {
                "*": {
                    "identifiers": [[8192, 3072], "mlpX.c_fc2.weight", "joint_transformer_blocks.2.ff_context.linear_2.weight"],
                    "file_256": ["ce3e475246258b94ee9dcb8b83292cb34edfffc2bbde46c74604d9c6cd7c585c"],
                    "layer_b3": ["cc6d383576c35a9709798d2e2b9e3eb31ba8c608040cf3712bc37871cfd14e21"],
                    "layer_256": ["3c13e6a965d03a49227d8b1606ba6a343a23772d8768407cc78d4ddb9102bc80"],
                },
            },
        ),
        (
            "info.dit",
            "hunyuandit-v1",
            {
                "diffusers": {
                    "identifiers": ["extra_embedder", "model.blocks", "skip_norm.weight"],
                    "file_256": ["4fb84f84079cda457d171b3c6b15d1be95b5a3e5d9825703951a99ddf92d1787", "e01db5e129e8ca1117e9cf473fc5a2b096949f03ab90048aeabbc328de7ec800"],
                    "layer_b3": ["aead6b61b17ebc77c4c186a4b82c193f11ec267b20d909726422ee9852e2e0b2", "885a056b94f6f9844c0660be489844d63bb74cc13316f441d10968fff3dd3120"],
                    "layer_256": ["d4842ce2b7f927203326b25ff4d6738ec9a8b95327f06791c387e4a351ed6ed0", "5af943f96f5dc9fecb1e92fe2b1fa17c94dd6947690201f4a5ee1a4a2721a68e"],
                },
            },
        ),
        (
            "info.dit",
            "lumina-next-sft",
            {
                "diffusers": {
                    "identifiers": ["time_caption", "feed_forward"],
                    "file_256": ["371153b7c7b7a64899d4016970c7cc472039f9c9b21ebe073adf0b8525cdf1bd"],
                    "layer_b3": ["fa134efd6e9672e7de2965e4895fc58879bd0a6c4fdf9165c278f2748254675f"],
                    "layer_256": ["3938a85568d9df186923edf04391d79e89e6199123bc175afb520e0948d1ae05"],
                },
            },
        ),
        (
            "info.dit",
            "pixart-sigma-xl-2-1024-ms",
            {
                "*": {
                    "identifiers": ["adaln_single", "scale_shift_table"],
                    "file_256": ["c34b520ef473329b945c2a21083cdf1337c5a468d23b3215b65576789bfd0305"],
                    "layer_b3": ["a199930ff537994872da77391955f0dd52eddd22ab9105388f0c5852f1b8021f"],
                    "layer_256": ["e0afd203aff5a1d192e325d0f59361373273d85d138b51768c3f10a75c154dc0"],
                },
            },
        ),
        (
            "info.dit",
            "pixart-xl-2-1024-ms",
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
            {
                "*": {
                    "identifiers": ["model.diffusion_model.joint_blocks.", "transformer_blocks.21.norm1_context.linear.weight", "transformer_blocks.31.norm1_context.linear.weight", "blocks.11.ff.net.2.weight"],
                    "file_256": ["ffef7a279d9134626e6ce0d494fba84fc1c7e720b3c7df2d19a09dc3796d8f93", "11fe06e22364b823dfeedc275912336b932b32a293a0b2f35ffac071990cc4de"],
                    "layer_b3": ["e411016545785046810b29cc3999f40bc6392be134a1318386c6f1c48f98726a", "a81e07ee67bc627e8b3c5e292ec1ca239009517a2106e8249d670ced0a88f746"],
                    "layer_256": ["13c982a6dc82d21c9f459e837d8c6f6d4696fd6e7e7b5783bdd2250b1f4fec61", "6ee79050373337bf63ac20916596df778bb22022bb38af986128a7459eda1463"],
                },
            },
        ),
        (
            "info.unet",
            "stable-diffusion-v1-5",
            {
                "*": {
                    "identifiers": ["up_blocks.3.attentions.0.transformer_blocks.0.norm3.weight"],
                    "file_256": ["6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa"],
                    "layer_b3": ["909c6ff3192ab2767e789a6125865bc23163db467ab78b1c633bad46a4293fad"],
                    "layer_256": ["ece771354ad470a82d56eda413ae3dd6c00d2de28ab3c56a88201d08d4424b4b"],
                },
            },
        ),
        (
            "info.stst",
            "t5",
            {
                "*": {
                    "identifiers": ["encoder.block.0.layer.1.DenseReluDense.wi.weight"],
                },
            },
        ),
        (
            "info.stst",
            "umt5",
            {
                "*": {
                    "identifiers": ["encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"],
                    "file_256": ["decf9b70814ed5e9965bfca9fbd0483462e2bf743790663025b7742f8c014c72", "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6"],
                    "layer_b3": ["1c943dbcb8b328a7c6c852921ddaefbd84c9df8c83bc51fe303c1f06cb734102", "1639a6467af0db1e15828d33b878e568cba1335947eeadd481170bcdc9ba8e33"],
                    "layer_256": ["58deeef888d4ded4ffababfbf8da27227a4a6ff8adfa42016e12c0180f713816", "178ebd3fa3418d33a2e45a80d8b9d3662ff4a8e75f3de3f0332f82c505d8152a"],
                },
            },
        ),
        (
            "info.stst",
            "mt5",
            {
                "*": {
                    "identifiers": [[250112, 2048], "text_encoders.mt5xl.transformer.shared.weight"],
                    "file_256": ["0524484ec81425ba9deef6fac1393a78ba9b1c9bfed704a4be5f9c7255975cc1", "32f70f1d187e131a5fc3e4f0edc97ce89360d8e2f1d90177a443a05296097acc"],
                    "layer_b3": ["a1d616c37711ec7b9073d04734af2f5fd02f9035a322eb46efeace922e104c51", "bc71d4259f4feaa0fb27c1f288765004840f39247cddc98b3ac37329ff1354d0"],
                    "layer_256": ["bd337daf0c1aa36896013109b406a0580aa3bb8ab9291d89df3015d737358e95", "2e40c48c96fc7df636aad96d3e78ed0ba9f68c3059e21b7fcf917f284c569a61"],
                },
            },
        ),
        (
            "info.unet",
            "kolors",
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
                    "file_256": [
                        "673b3173b037fb5f65b14fde37267390641a36726683de75dcf9df76fce2b866",
                        "45c1eb5ce9b69efac891ad459b15c215cd90a986adbbfaf3effd3a89578cbcaf",
                        "088ddf1e444abf399007b2da2bac87791df165c69f477994f6b3c745a20904b0",
                        "39cec96c7212607f9e526db719bf1df507166d09f4748676c13b0d31cd4adb07",
                        "31ffe2f1a3e2351d658fc7d3002a4eca22466a680f7fb3715b1e3768476f9633",
                        "dfe24009fc881011f350d08d9d13be13a1a3b3cbfed667435efe0fd419aca099",
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
                    "file_256": ["4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7", ""],
                    "layer_b3": ["261559c8eaccae558f72621804a9ee188d338e45e2c622a58db709ac190198ba"],
                    "layer_256": ["3db58cf834d2f81abb1e035131956da4c90451074c681d0db10810e55e60c2c4"],
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
                    "file_256": ["9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a"],
                    "layer_b3": ["c65ba812ce3ce056eb1585673f62fb896afe6ec049faaf00a97bc35c9a398c44"],
                    "layer_256": ["79c07e339865fe9e22c80f723d728c778130acd07a330339c68218b92bb7b3b8"],
                },
            },
        ),
        (
            "info.unet",
            "stable-cascade",
            {
                "decoder": {
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
                    },
                    "file_256": [
                        "fe92687deefcfb33bb3ec181254b55fe4e434c5084ce9d38815eaa32487ad376",
                        "2c8d58b267678aecfa6705a0a0375c88613065a8a8d32ad3a4c3867f5461cb3a",
                        "6c218dc948575e3b14b03dffe2014d7870ac505005770ce3abdc28e920a03c05",
                        "a6c3d534a9be308e95d2c3224af94a854bebd9b503f620f1ae3c8e6ba4a341bf",
                        "7b431ea7d0f10e72b3eaece353bf6bf2f6bc717b6f4207411be186b40dec1f43",
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
                }
            },
        ),
        ("info.aet", "wavlm", {"kokoro": {"repo": "hexgrad/Kokoro-82M"}}),
    ]
    try:
        assimilate(mir_db, data_tuple)
    except (KeyError, StopIteration) as error_log:
        print(error_log)
        pass


def auto_supplement(mir_db: MIRDatabase):
    """Create MIR entries missing from the database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="refiner-1",
            repo="stabilityai/stable-diffusion-xl-refiner-1.0",
            file_256=[
                "54f9cd2f2daf3aeec0b2708fa3dbc0e84e4f8ddd1ddead42e5bc60c6572c989f",
                "7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f",
                "3ea0376dcf065eaefd27806394a90e310001b1a71d4f1cf1f655e86c0e566ffe",
            ],
            layer_b3=[
                "6281355dbb37e5769c9460ae0ac75506d89932e2f97b09d9ade32ecf191e75ba",
                "afb0639aae2eb65577c12d4a30cf7c9b3620ae63ba64a8fa632b58608c8a7a2e",
                "669046014b69d98ab0f6fbb59547644436e0275f8b638f467ce2a873c3313683",
            ],
            layer_256=[
                "bb9eadbfabb52c0d8645783525a3fa70b59e9d7d09d5290d742a303262e793a2",
                "c5adb56fe51343af2c3d493eb9f41515c204bd91eb9f40b983d45f70a1fa3b6d",
                "1f838e39ed6e916258aee6990b72c09b34aa8eb3b5342234a497b8852b3df1c6",
            ],
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
            series="stable-diffusion-xl-1",
            comp="pony-diffusion",
            file_256=["499513276a79a4e8c6d601248eb67178d5f3322c92ac8cec1f9b40f6927d2242"],
            layer_b3=["572ae32fb0ae0d14d259f1de7250dee16fb17434208780ca0560de41596720a4"],
            layer_256=["d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl-1",
            comp="animagine",
            file_256=[
                "8ece83aa1bed1fb39a2b81f1660f0ce6889218e493c1f2ed55e9f15f59a7e03f",
                "6327eca98bfb6538dd7a4edce22484a1bbc57a8cff6b11d075d40da1afb847ac",
            ],
            layer_b3=[
                "268ffbb120670b9c4b25158bd474c787740884b7738b48203aa03c4c3f00028f",
                "18fda1a55cad137d62c81d4328f5ece85d88b126261e06b9e14ab68055d5d484",
            ],
            layer_256=[
                "c21d1c38813e078817122e12866ab39f5aa7f56945dd4a8beee3cae1e0f139e7",
                "b916c162c981155aaf74e93d5314038af6767bb5a129c51ee05a1fb6a206c6ac",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl-1",
            comp="illustrious",
            layer_256=["c4a8d365e7fe07c6dbdd52be922aa6dc23215142342e3e7f8f967f1a123a6982"],
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl-1",
            comp="playground-2-5-base",
            layer_256=["a6f31493ceeb51c88c5239188b9078dc64ba66d3fc5958ad48c119115b06120c"],
            identifiers=["edm_mean", [1, 4, 1, 1], 2516],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl-1",
            comp="playground-2-5-aesthetic",
            repo="playgroundai/playground-v2.5-1024px-aesthetic",
            file_256=["bcaa7dd6780974f000b17b5a6c63e6f867a75c51ffa85c67d6b196882c69b992", "956dca99114aaa5c3eb526381309d37ee96737e78ed64c8ae613409f47c3f65a"],
            layer_b3=["7e9be9bd9a3aed1ad7207e2f77c98c24c3a75f6adcc9b53514033c6c3365d289", "5c6dfcc8d01dfb64723f8f5785caa080e2987859c0a050470bfdbe5312be9efc"],
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
            file_256=[
                "a5b04df4072698395387c21e8da0176d03f6557e0c38ff1dd3bf469ebab9d0fd",
                "a91b46de2055b3511ee87523b57862648856e8c00100161d5b520543a7302755",
            ],
            layer_b3=[
                "4dd3174edf6b680ce9daf3de643e33ae2c4f09a4d5968da61ea48885f3a193c0",
                "9fdf191b2c58b2a6e190396e12314530593dca4f2a2bee389ec5175da5e52af8",
            ],
            layer_256=[
                "14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443",
                "7ce8d449b32a9c959431ade729b513ee7a6457f11e1c13e3ef04dd8db3494621",
            ],
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
        mir_entry(domain="info", arch="stst", series="silero-vad", comp="*", repo="onnx-community/silero-vad", pkg={0: {"onnx": "onnx"}}),
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
            file_256=["4236455adeaeb4ed444d63b253ec99805022d17e962ed7261ada9c72ce11cfee"],
            layer_b3=["c1a6f83585398fe452d20596a79a522e2986f4c2c01a40e7bfd787af113735d3"],
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
            file_256=[
                "565cb2487351282e8e4dbeb88e63f4ad28217ce0439f5a8e6525a924807d2d9b",
                "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
                "4f2751ceeb2a96edd693e539dc5d6bba0b8d3814f49a9b3798403a0cec4b2e3d",
                "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e",
                "83690f3cc37cecb5e907f41ab0f7abb0855ef24a0a8aab9259f2888ce85a34e2",
                "8b28110e64e0019489aaa54ed85ee9f409ce7aa22e3bfdcbd980acd65e7c2eae",
                "a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00",
                "8490f7a22615c20651a63dbe7b4241929826a4de20292dc8e63bfc3c61e3654f",
                "32f70f1d187e131a5fc3e4f0edc97ce89360d8e2f1d90177a443a05296097acc",
                "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
                "74d5ecd5cba5494a2e78ec45b5770e25d3f3cb1f9864ec59ec85ff140de3a8d0",
                "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",
            ],
            layer_b3=[
                "ca94e03b7b1fdcb0d6ff5205eac56f145d2dff8a9c489faf80935bfec8387f18",
                "c0e2b054bedd782909191b05748a88c28d1538fa91789fec63f036ba01dcc001",
                "672de9b79d14001de7d1109ffc52e4d0cccc3bfee6f45648fa347703b58e2b99",
                "abdb187a996c51cb0469630c124b14eeb0bb8f5f635aca6c71dea264f8bd61ae",
                "8926f862b7763fd9688af317eba7809aa71a478484be0c738c269de368ace4a7",
                "3e1a6461b88ab45defd940a12935ada6d3655fcc7979da773f705896e99884a9",
                "66e35f8419a7d13f15a2e0fd8711498bb7586e57da15dbc858e45b96634d9f7d",
                "e616b754cf55e55b3f9f17ab7e1fff95f0607c81782822fc1223ae22fb1e9f36",
                "b79e5f1878a62cd726bb4f9fc1415cacb071d278440e9026290c7b36cb41e1d4",
                "c6325f87ead2e0ad1dee089c68ce5e7e8b5ff2c78f5882a8758115b450e7c8ef",
                "3f4e51dec6d542759cdea49b3bec14c090a4908f953fa3e182e2ea43b5b05402",
                "52270564847ec3972707b9aea5b5d618fe255ff11434a92f3f3e46a486cde3e1",
            ],
            layer_256=[
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
            ],
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
            file_256=[
                "cb0cba1ead482a850532ebe5ff6b5c8d4456aee32a5228acf0a31e7d9472415e",
                "39e79c916feca4ddf546d9fe923e664714b59ea61074f7228037d17c302f3d17",
                "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082",
                "778d02eb9e707c3fbaae0b67b79ea0d1399b52e624fb634f2f19375ae7c047c3",
                "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
                "71e183d11db0c6b6282a4d9e0abb74125edc8692393e89ed8ee5571005f35cb1",
                "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc",
                "87c1c0b0894c9e9e10b962e597e8d64dd3a3a2d372c389922b335a53c250b2ae",
            ],
            layer_b3=[
                "f58a22a381f79985b6d38782f6110a52c2f319b40fdedd3b88b24945dfcbdf64",
                "8faa00b8fd1dbd9286a7237df18caeb8c91af100a6813849b6bae272a01dd7b7",
                "ab5bebc98299c155251a06deccde599ba0128038ee3ce021e8c59a45f58f72c0",
                "c70e9d86a9dcbbbe7c269ef9dfac96ce9c96c46922577338cc1902e5fe936315",
                "f285e9b7b70745df81adc8b558ec74b536b79b6fc02a453ecc61ea9d13f25f1a",
                "7ab17bfa06ab8d65840997ef641f3f593d096860e20141f1eeb0169d131c1c23",
            ],
            layer_256=[
                "48daa3d8f939972e69f044533a4312a941971c18c78255f5e555fa26faf664c1",
                "60f5734a74c342be8b0011fc704e718431839790bcfdc7d7004fc39d70f7fec6",
                "6e76e25b4a55dddfa2eecf4b7ab189a8148658a9f6df165c00170f6ce661033c",
                "2d5249df489fec9137cc3a5e9bda499dd9b72a957ddd8e7ad4e99ff3684bad99",
                "3bf085e701713ed3e79775dafea375c3e2a43659ad1ee788b1b393c0aeff9f0e",
                "efb7976800692772e449c81a739339f59394886590ff3f768b0f9ddd87d2a94c",
            ],
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
            file_256=[
                "ca18e0c67c1ef1e64cac22926266765b60688f692307ecc06283d987c5768134",
                "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
                "fa5b2e6f4c2efc2d82e4b8312faec1a5540eabfc6415126c9a05c8436a530ef4",
            ],
            layer_b3=[
                "d754db276f2d89d2808abb7086b3b8eccee43ac521c128d21a071f3a631474a8",
                "2eb93685b34719e1d1e0541d8902b0a592d95848f80657e32816cf3b152a0f31",
            ],
            layer_256=[
                "270e998633eb22145100a3889a62ca270d5080654735e5ff8dda09a7c233af8d",
                "df18800c2a9d9318c4323d991a0fb24a6a9afceb41bea203812f60517c301536",
            ],
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
            comp="stable-diffusion-xl-1",
            repo="ByteDance/SDXL-Lightning",
            pkg={0: {"diffusers": {"load_lora_weights": {}}, "generation": {"num_inference_steps": 4, "guidance_scale": 0}}},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="pcm",
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
            repo="alimama-creative/slam-sd1.5",
            pkg={0: {"diffusers": {"load_lora_weights": {}}}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
            series="stable-diffusion-xl-1",
            comp="turbo",
            file_256=["a599c42a9f4f7494c7f410dbc0fd432cf0242720509e9d52fa41aac7a88d1b69"],
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
            comp="stable-diffusion-xl-1",
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
            comp="stable-diffusion-v1-5",
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
