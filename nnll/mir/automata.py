# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""自動化索引"""

from logging import INFO, Logger
from typing import Dict, List, Tuple

from nnll.mir.maid import MIRDatabase
from nnll.mir.mir import mir_entry
from nnll.mir.tag import make_mir_tag

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

# b3 hashes
# SD3.5 LARGE 878387
# ['EBAA8DB98ADFBC460A59BDBAB8979E5DBE9ABAAF10973E7FF6BA7B6D4ABBF7CB']
# SD3.5 MEDIUM 896953
# ['FEA996B8E95723AA902AA14D6F44FB1B4F61FF5AF8581431CDE2540D4FA7F9F3']
# AURAFLOW 0.3 573326
# ['5D0E864E51E1747B5CE28C4368EEA6D858162AA62E155656DE49833EB8055217', '2C0F0988184E9E7CB5E4BA16A31C03073CBBD49BE5B99982D3D250E7BB511083', 'FAEA311A0D9F2BCE77B27D689BE33F0F3147F37C6A47170D0D5631A338BD9ED1', '17D652AE4ECF5404C86CBE895D75504D5029C9E27047B95861E1E3CA9C07B8B0']
# SCHNELL 699279
# ['202D1D57667FDC0B5B088A46DCE7C801FFCF2E0933D4C521E272399ECE17AA7B','79C38CA0E3DB84E77849DC8DDCD0A6AE4DEC34843842628F25C5A4E0AA2756E3']
# DEV 618692,699279
# ['40ACE2CF5D03A95EC710054EC5E4C8E864341A7EBFEFE188ED750C41A0463795','0B9CB7749990B8E3E26E46FF5DC2CA18AA5F970DA68E8AEA681EE995898D2542']
# ANIMAGINE XL 4 1188071
# ['C8FACC6833DCF4FA0E3194539519EC2AD6B10C274605DCD988BBBDF160223A76', '89EA185CBCE8F6C0AD0308A92055C5C50002BA20803EEC42142566763AA7C0C2', '2A4A32D2E607AAA5AFCECC13F3F0035B21C949EF1DE7C7DC7836AFF11CA65E5B']
# ANIMAGINE XL 3.1 260267
# ['DE1931B2EEF4D32725B3064FE3F50AB5392E78DDF497C8A1B6BCF175CFE6C73A', '9EF46CD2F47F064860494C1D6E6BBB0D1A12052B641ED09C78DD265F767CD110', 'BA8B3384D287A2AE24E59D05898F92A39CCD56C28CD5560F854CF0900FBF8948', '49DF88B1702C2C1C684AC01D40CE0147934FBC9245DE676EA307812231B4766F']
# ILLUSTRIOUS 795765
# ['EF3A6DD9C8A71E07D77CE8E84780524357E629EE8750EB3018AED648D4C803F1']
# PONY 257749
# ['E0BB278C0127A4AC4267498174E0226D6A0A77A636BDF82B770CF05D5B85D3CA', 'D46EB6988A26BCAF9E9CFA5E5C6264C4EE1A70F2018F33B8BC2DD7CA0681B490', '65B4E5A1E83C6A0133C1981C464E2F2D1ACA7ADE10BBC661C3C9EA63A848FE06', 'D95F9AD55421D99EF9F39D9A85CF91366514737B26BE52E467A328517C645B94', '0501A74F91EAA4D67AF7B25F8C8AF2870CF0A28B99438D9F4B4818BA43C3D3EE', '58C5003E9151020DCC66587EF0A69F8806812CB3E270D6F7E57FCF9BD9E73CCB']
# PLAYGROUND V2 1024 AESTHETIC FP16 307370
# ['9910B7A72C69CDDDF924966318C3187312F29E46F7D1006BAAF8CB5F3B188EAA']
# PLAYGROUND V2.5 1024 AESTHETIC FP16 325263
# ['5FD2BAEA87F36F0597DE81312EF6F6ED3EADED23A54498CA77CCE02A98532E02', '78318DF13E0A01C69E9AFF5C65171CDA03CE955B19BAFD185989EDF1FD8E6570']
# HIDREAM 1562709
# ['E42D95FF782D391DFB8B561C37ECD1DF60CE1A6E97DF4AC5564E3BBD8DC002F2', 'A4DBD21D675F9A3012F86747CCF7D52A9EE0A43A4DC72E2B7A08916F4639BC90', 'F19A3D6987C9884E5521E35C715B12FFEEFDD5AF4C918436037C1C89E83F6FDC', '34527337B85AB2272120C8C2181D5864DFBB3B203B72938E0DDE18387FEBE1BF', 'A588FCC0CD75FCD38CE2B68352632A8607349699F3968F5F46E60D0EF73F5E6A', '8F40C3BDC5D2F2C0388B8364F454D962A2605B6D2628F40AB275BBFF9610562D', 'E5DDEC1C287B048D9E07C655895511BAA68C3A5C0F4E476E149FDACA3281AD98']
# WAN VIDEO 1329096
# ['65AFB17E7F1F28F75A02BFF5F85AF2F83875647A8392C13E5F64AB28D772B4B7', 'E731BF9FB7614C712674F678A6EE60726A2A1A471CFA3BAD3106E925C138F843', 'C3AFF07CFF3A90224C4742FD8D0323987C99D1BB13AAC472AADF72641D2594AA', '0F84C6D87440CD058B6ACD525774C63367C10EBCEC911C817D68390ACCC11A5C', '822D712DA6C4F99567553F5409D7EBF78842CE2DF4F5B0C01E2A159B7E3CBB88', 'DE4037BC9D3AED3FCC081FD95EC398C04D54B06BC79A6AE51FA9BC574BF6D63B', '08FC8A4DEDD056A24ACABD1BE42092E61B806056B72FCBDB3F8A98F1C3B9F8CD']
# CHROMA 1330309
# ['0440A97DA1DD918F1FBDAB06508A139C4B9195DE1978737AAE19374404B2B62E', 'FDD6AF4C10B10A026FD0F1245E947D360A8C0E78B889EC9E77D1A1B8CB2EA339', '94810CA30128722B5DD3E55B9D61CFE404ACEBFCA311BB099DEB68E6942B4D60', 'FF6DC9C5B6951654DDF898A52F929790334B40A2FF2AC5627BD90919CC7B3C0C', 'E23677576F35E818699711644A61BA7510DB305809279ED3612C588CF6B52E39', '3EC4AA2EED939FB534E65544173185C22EF41158FA0D025B91B9F81B38630A51', '5CAD59F110A15AD6DCF334596C769C91EBA533093B26A30EC32C1EDE55A18D46', 'CE21CB76364AA6E2421311CF4A4B5EB052A76C4F1CD207B50703D8978198A068']
# KOLORS 566526
# ['0FEFC290666FCBE2F09E0838C39AF44C07903AAC314C77E454C49A6DEAD44A2B']


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
    from nnll.mir.indexers import create_pipe_entry, flag_config
    from nnll.mir.tag import make_mir_tag
    from nnll.tensor_pipe.deconstructors import root_class

    def re_create_pipe_tag(repo_path: str, class_name: str, addendum: dict) -> tuple:
        if any([source for source in ["google", "microsoft"] if source in repo_path]):
            annotations = root_class(class_name.replace("Model", "Config"), "transformers")
            mir_prefix = flag_config(transformers=True, **annotations)
            mir_series, mir_comp = make_mir_tag(repo_path)
            mir_prefix = f"info.{mir_prefix}"
        else:
            mir_series, mir_data = create_pipe_entry(repo_path=repo_path, class_name=class_name)
            mir_prefix, mir_series = mir_series.rsplit(".", 1)
            mir_comp = list(mir_data)[0]
        return mir_prefix, mir_series, {mir_comp: addendum}

    details = [
        (
            "stabilityai/stable-diffusion-xl-base-1.0",
            "StableDiffusionXLPipeline",
            {
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
                "file_256": [
                    "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",  # aio
                    "e6bb9ea85bbf7bf6478a7c6d18b71246f22e95d41bcdd80ed40aa212c33cfeff",  # aio vae 0.9
                    "357650fbfb3c7b4d94c1f5fd7664da819ad1ff5a839430484b4ec422d03f710a",  # diffusers
                    "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139",  # fp16 diffusers
                ],
                "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
                "layer_b3": ["8be44fa13c1efa60f8bcadaa57f1d718473f9660f03c4f0e65dc037960d8cba1"],
                "identifiers": ["logit_scale", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight", "add_embedding.linear_2.bias"],
            },
        ),
        (
            "lodestones/Chroma",
            "ChromaPipeline",
            {
                "pkg": {
                    1: {
                        "generation": {
                            "neg_text": "",
                            "num_steps": "28",
                            "latent_size": [64, 64],
                        },
                    }
                },
                "file_256": [
                    "53adcb3b6b6005758d40e2d8058b044ed4892bc8616efb7a62cc2dd384be07de",  # v1
                    "2c41e8a9831f3be1eaff2c2ed590abb62e4534e814f7ec58a5fd74ff71dc2036",  # v46,
                    "0a7b2d9699dbd22b3744ee2692900cabcfb731a43dac13729c33807f2bb7c9f6",  # v37 detail
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
                "file_256": ["ce3e475246258b94ee9dcb8b83292cb34edfffc2bbde46c74604d9c6cd7c585c"],
                "layer_b3": ["cc6d383576c35a9709798d2e2b9e3eb31ba8c608040cf3712bc37871cfd14e21"],
                "layer_256": ["3c13e6a965d03a49227d8b1606ba6a343a23772d8768407cc78d4ddb9102bc80"],
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
                ],
                "layer_b3": [
                    "aead6b61b17ebc77c4c186a4b82c193f11ec267b20d909726422ee9852e2e0b2",
                    "885a056b94f6f9844c0660be489844d63bb74cc13316f441d10968fff3dd3120",  # distilled
                ],
                "layer_256": [
                    "d4842ce2b7f927203326b25ff4d6738ec9a8b95327f06791c387e4a351ed6ed0",
                    "5af943f96f5dc9fecb1e92fe2b1fa17c94dd6947690201f4a5ee1a4a2721a68e",  # distilled
                ],
            },
        ),
        (
            "Alpha-VLLM/Lumina-Next-SFT-diffusers",
            "LuminaPipeline",
            {
                "identifiers": ["time_caption", "feed_forward"],
                "file_256": ["371153b7c7b7a64899d4016970c7cc472039f9c9b21ebe073adf0b8525cdf1bd"],
                "layer_b3": ["fa134efd6e9672e7de2965e4895fc58879bd0a6c4fdf9165c278f2748254675f"],
                "layer_256": ["3938a85568d9df186923edf04391d79e89e6199123bc175afb520e0948d1ae05"],
            },
        ),
        (
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "PixArtSigmaPipeline",
            {
                "identifiers": ["adaln_single", "scale_shift_table"],
                "file_256": ["c34b520ef473329b945c2a21083cdf1337c5a468d23b3215b65576789bfd0305"],
                "layer_b3": ["a199930ff537994872da77391955f0dd52eddd22ab9105388f0c5852f1b8021f"],
                "layer_256": ["e0afd203aff5a1d192e325d0f59361373273d85d138b51768c3f10a75c154dc0"],
            },
        ),
        (
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "PixArtAlphaPipeline",
            {
                "identifiers": ["aspect_ratio", "y_embedding", "emb.resolution", "caption_projection"],
            },
        ),
        (
            "stabilityai/stable-diffusion-3-medium",
            "StableDiffusion3Pipeline",
            {
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
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "StableDiffusionPipeline",
            {
                "identifiers": ["up_blocks.3.attentions.0.transformer_blocks.0.norm3.weight"],
                "file_256": ["6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa"],
                "layer_b3": ["909c6ff3192ab2767e789a6125865bc23163db467ab78b1c633bad46a4293fad"],
                "layer_256": ["ece771354ad470a82d56eda413ae3dd6c00d2de28ab3c56a88201d08d4424b4b"],
            },
        ),
        (
            "google-t5/t5-small",
            "T5Model",
            {
                "identifiers": ["encoder.block.0.layer.1.DenseReluDense.wi.weight"],
            },
        ),
        (
            "google/umt5-small",
            "UMT5Model",
            {
                "identifiers": ["encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"],
                "file_256": [
                    "decf9b70814ed5e9965bfca9fbd0483462e2bf743790663025b7742f8c014c72",  # fp16
                    "0a07449cf1141c0ec86e653c00465f6f0d79c6e58a2c60c8bcf4203d0e4ec4f6",
                ],
                "layer_b3": [
                    "1c943dbcb8b328a7c6c852921ddaefbd84c9df8c83bc51fe303c1f06cb734102",  # fp16
                    "1639a6467af0db1e15828d33b878e568cba1335947eeadd481170bcdc9ba8e33",
                ],
                "layer_256": [
                    "58deeef888d4ded4ffababfbf8da27227a4a6ff8adfa42016e12c0180f713816",  # fp16
                    "178ebd3fa3418d33a2e45a80d8b9d3662ff4a8e75f3de3f0332f82c505d8152a",
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
                    "bc71d4259f4feaa0fb27c1f288765004840f39247cddc98b3ac37329ff1354d0",  # fp16 enc
                ],
                "layer_256": [
                    "bd337daf0c1aa36896013109b406a0580aa3bb8ab9291d89df3015d737358e95",
                    "2e40c48c96fc7df636aad96d3e78ed0ba9f68c3059e21b7fcf917f284c569a61",  # fp16 enc
                ],
            },
        ),
        (
            "microsoft/Phi-4-multimodal-instruct",
            "Phi4MultimodalModel",
            {
                "pkg": {
                    "0": {"transformers": "Phi4MultimodalModel"},
                },
                "file_256": [
                    "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6",  # mini
                ],
                "layer_b3": [
                    "b391ba867f6074f488f39bef52a475bf4e983328d5d437d196f2882cae79620f",  # mini
                ],
                "layer_256": [
                    "354276af1e65f68606c5cca62b4fc1ec87e905fc3a858aa30d01fe39d5a1d5d0",  # mini
                ],
            },
        ),
        (
            "Kwai-Kolors/Kolors-diffusers",
            "KolorsPipeline",
            {
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
            },
        ),
        (
            "stabilityai/stable-cascade-prior",
            "StableCascadePriorPipeline",
            {
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
                    "673b3173b037fb5f65b14fde37267390641a36726683de75dcf9df76fce2b866",  # lite bf16
                    "45c1eb5ce9b69efac891ad459b15c215cd90a986adbbfaf3effd3a89578cbcaf",  # pretrained
                    "088ddf1e444abf399007b2da2bac87791df165c69f477994f6b3c745a20904b0",  # stage c aio
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
            },
        ),
        (
            "black-forest-labs/FLUX.1-schnell",
            "FluxPipeline",
            {
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
                        "precision": "ops.precision.bfloat.b16",
                    },
                },
                "file_256": [
                    "fe92687deefcfb33bb3ec181254b55fe4e434c5084ce9d38815eaa32487ad376",  # lite bf16
                    "2c8d58b267678aecfa6705a0a0375c88613065a8a8d32ad3a4c3867f5461cb3a",  # bf16
                    "6c218dc948575e3b14b03dffe2014d7870ac505005770ce3abdc28e920a03c05",  # b aio
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
    ]

    data_tuple = [re_create_pipe_tag(*entry) for entry in details]
    assimilate(mir_db, data_tuple)


def auto_supplement(mir_db: MIRDatabase):
    """Create MIR entries missing from the database"""
    tag = make_mir_tag("stabilityai/stable-diffusion-xl-refiner-1.0")
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=tag[0],
            comp=tag[1],
            repo="stabilityai/stable-diffusion-xl-refiner-1.0",
            file_256=[
                "54f9cd2f2daf3aeec0b2708fa3dbc0e84e4f8ddd1ddead42e5bc60c6572c989f",  # diffusers
                "7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f",  # aio
                "3ea0376dcf065eaefd27806394a90e310001b1a71d4f1cf1f655e86c0e566ffe",  # fp16 diffusers
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
    sdxl_base = make_mir_tag("stabilityai/stable-diffusion-xl-base-1.0")[0]
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_base,
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
            series=sdxl_base,
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
            series=sdxl_base,
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
            series=sdxl_base,
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
    repo = "playgroundai/playground-v2.5-1024px-aesthetic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series=sdxl_base,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "bcaa7dd6780974f000b17b5a6c63e6f867a75c51ffa85c67d6b196882c69b992",  # fp16
                "956dca99114aaa5c3eb526381309d37ee96737e78ed64c8ae613409f47c3f65a",
            ],
            layer_b3=[
                "7e9be9bd9a3aed1ad7207e2f77c98c24c3a75f6adcc9b53514033c6c3365d289",
                "5c6dfcc8d01dfb64723f8f5785caa080e2987859c0a050470bfdbe5312be9efc",
            ],
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
                "edm_mean",
                [1, 4, 1, 1],
                2516,
            ],
        )
    )
    schnell = make_mir_tag("black-forest-labs/FLUX.1-schnell")[0]
    dev = make_mir_tag("black-forest-labs/FLUX.1-dev")[0]
    repo = "shuttleai/shuttle-3.1-aesthetic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                0: {
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
            series=schnell,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                0: {
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
    repo = "jack813liu/mlx-chroma"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=make_mir_tag("lodestones/Chroma")[0],
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                0: {
                    "chroma": "ChromaPipeline",
                    "generation": {"neg_text": "", "num_steps": "28", "latent_size": [64, 64]},
                }
            },
            file_256=["6ddc9e2bbe3376ab5ee9f10b2d947f127b6bf6f879f06f316a2208bb0da357b8"],
        )
    )
    repo = "enhanceaiteam/Mystic"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev,
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
                file_256=["e61ae2f638f56ee07a37592cd5a6a9e7d642560ddc78a76ee4a7f96d6922f1be"],
                layer_b3=["ba07192df1860b0db8c4b46442d2c3712973a798a697ca5f688f544fe3bce303"],
                layer_256=["08255792456ab823304f66cdd6a1b90001d8eb3f646a90aebc414afe5259c94c"],
            )
        ),
    )
    repo = "shuttleai/shuttle-jaguar"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=schnell,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            pkg={
                0: {
                    "diffusers": "DiffusionPipeline",
                    "generation": {"guidance_scale": 3.5, "num_inference_steps": 4},
                }
            },
            file_256=[
                "dcbc4f2470b177eed12c7d7515c0e7342515a849ebd31a50c8d8d43913d7bd32",
                "26a7aa64c0798a3549e1d767932da0a7fb82b49f8edcbdcde804a20d9ed1478f",  # mlx q8
            ],
        )
    )
    repo = "freepik/flux.1-lite-8b"
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series=dev,
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
            series=dev,
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
            series=dev,
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
            series=dev,
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
            series=dev,
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
            series=dev,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "0407108e446a4f57efffc5e7518bc374876af970d3c6068dc4074de0d221c615",  # aio
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
            series=dev,
            comp=make_mir_tag(repo)[0],
            repo=repo,
            file_256=[
                "5d6dce30a266ccbf530c3a3bf253cd5486720a8fb71cdeed556c28304201dc2f",  # aio
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
    repo = "Alpha-VLLM/Lumina-mGPT-7B-768"
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
                    "inference_solver": {"FlexARInferenceSolver": {"precision": "bf16", "target_size": 768}},
                    "generation": {"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
                }
            },
            identifiers=["model.embed_tokens.weight"],
        )
    )
    repo = "google/t5-v1_1-xxl"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="stst",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"transformers": "T5ForConditionalGeneration"}},
            identifiers=[[4096], "encoder.embed_tokens.weight", "text_encoders.t5xxl.transformer.shared.weight", "t5xxl"],
            file_256=[
                "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e",  # shuttle/dev diffusers
                "565cb2487351282e8e4dbeb88e63f4ad28217ce0439f5a8e6525a924807d2d9b",  # bf16 aio
                "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",  # fp16 aio
                "4f2751ceeb2a96edd693e539dc5d6bba0b8d3814f49a9b3798403a0cec4b2e3d",  # fp16 diffusers
                "83690f3cc37cecb5e907f41ab0f7abb0855ef24a0a8aab9259f2888ce85a34e2",  # flux diffusers
                "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",  # fp8_e4m3fn
                "8490f7a22615c20651a63dbe7b4241929826a4de20292dc8e63bfc3c61e3654f",  # qfp8_e4m34n
                "8490f7a22615c20651a63dbe7b4241929826a4de20292dc8e63bfc3c61e3654f",  # qfp8_e4m34
                "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",  # q5 k m gguf
                "d8720addef2596fef86b1b22e4b62875c9118779ba8723759a75dfcbc649ffd5",  # mystic mlx
                "7d0eac95abe8daae454bcd3d166b8bfc6a35fe68278f97479d62dbb6850f38c0",  # mlx flex2
                "ceabd6f71c7112cfaa4dfca8711dda97b79fb9b25983f1c95532de226045f1f8",  # shuttle jaguar q8
                "49e139f50824fef40908ef4307c851e7adaa8b91bed44054c4829600dbedfdda",  # shuttle 3 q4
                "211ade1d474f5dc83190aec8be5c4baf52643777790d64de0cbd84f63613e5e9",  # flex1 q8
            ],
            layer_b3=[
                "ca94e03b7b1fdcb0d6ff5205eac56f145d2dff8a9c489faf80935bfec8387f18",  # bf16
                "c0e2b054bedd782909191b05748a88c28d1538fa91789fec63f036ba01dcc001",  # fp16 sd35
                "672de9b79d14001de7d1109ffc52e4d0cccc3bfee6f45648fa347703b58e2b99",  # fp16 sd35 diffusers
                "abdb187a996c51cb0469630c124b14eeb0bb8f5f635aca6c71dea264f8bd61ae",  # shuttle 3 aesthetic diffusers
                "8926f862b7763fd9688af317eba7809aa71a478484be0c738c269de368ace4a7",  # diffusers
                "e616b754cf55e55b3f9f17ab7e1fff95f0607c81782822fc1223ae22fb1e9f36",  # fp8 e4m3fn
                "b79e5f1878a62cd726bb4f9fc1415cacb071d278440e9026290c7b36cb41e1d4",  # fp8 e4m3fn sd35
                "3f4e51dec6d542759cdea49b3bec14c090a4908f953fa3e182e2ea43b5b05402",  #  q5 k m gguf
                "77619d5278d9f547ddac17d4d99df56cb6a3a9e660ae31b2f896a4297907e62e",  # mlx t5 jaguar
                "c87c9d3cc7becc46ee34821299cf8551a6df5541582a45469a031bccdc4bd340",  # mlx shuttle t5 q8
                "a49c2bc301733967ddff113790e301773dc5dd71368b657af4141458de593ced",  # mlx flex2 preview
                "c2ea94030ea362e03d73d448fa5353ace0a449dc38c51a4a49fb148444ebb8ef",  # mlx shuttle3 diff q4
                "4a90463350f08ef41479da1d561ab41b8f8b792f1603a092226a838156aebfb0",  # mlx flex1 alpha q8
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
    repo = "openai/clip-vit-large-patch14"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"diffusers": "CLIPModel"}},
            identifiers=["text_model.encoder.layers.0.mlp.fc1.weight", "clip-l"],
            file_256=[
                "cb0cba1ead482a850532ebe5ff6b5c8d4456aee32a5228acf0a31e7d9472415e",  # long vit best
                "39e79c916feca4ddf546d9fe923e664714b59ea61074f7228037d17c302f3d17",  # vit l detail improved hit gmp
                "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082",  # shuttle 3 aes
                "778d02eb9e707c3fbaae0b67b79ea0d1399b52e624fb634f2f19375ae7c047c3",  # playground 2.5
                "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",  # playground 2.5 fp16
                "71e183d11db0c6b6282a4d9e0abb74125edc8692393e89ed8ee5571005f35cb1",  # sd3.5 fp16
                "5c3d6454dd2d23414b56aa1b5858a72487a656937847b6fea8d0606d7a42cdbc",  # sdxl diffusers
                "87c1c0b0894c9e9e10b962e597e8d64dd3a3a2d372c389922b335a53c250b2ae",  # L
                "bd289dd57fee86bc8816b55919a2b03f9c3c75af6025e21777325a6730872325",  # jaguar mlx
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
    repo = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="vit",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"diffusers": "CLIPModelwithProjection"}},
            identifiers=["31.self_attn.k_proj.weight", "text_model.encoder.layers.22.mlp.fc1.weight", "clip-g"],
            file_256=[
                "ca18e0c67c1ef1e64cac22926266765b60688f692307ecc06283d987c5768134",  # seaart furry g
                "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",  # aio
                "fa5b2e6f4c2efc2d82e4b8312faec1a5540eabfc6415126c9a05c8436a530ef4",  # playground 2.5
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
    repo = "ByteDance-Seed/BAGEL-7B-MoT"
    series, comp = make_mir_tag(repo)
    mir_db.add(
        mir_entry(
            domain="info",
            arch="aet",
            series=series,
            comp=comp,
            repo=repo,
            pkg={0: {"Bagel": "app"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp=sdxl_base,
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
            comp=sdxl_base,
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


def auto_audio(mir_db: MIRDatabase):
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
                    "stage_2": {
                        "transformers": "AutoTokenizer",
                        "generation": {"return_tensors": "pt"},
                    },
                },
            },
        )
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
                    "stage_2": {
                        "transformers": "AutoTokenizer",
                        "generation": {"return_tensors": "pt"},
                    },
                }
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
            file_256=["5a5cb3d87478f2e74dfca208ee52209ccfce024095e137097fd276026506e45f"],
            layer_b3=["3e9b5017cfe67a7804ac717b18b6add42ffc0bd3353490df2bcc520eaaef79b6"],
            layer_256=["dbedf0e2115aa309b92689f86534be4a77b91d7900365e1717879fbb19b849f6"],
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
            layer_b3=["41ca5931452b3ffee588c6c7e5bd327c4e914141604eaf3fd05f4a790ac83bb2", "7dc736cd5d840182792bde4edfbf5ddc5aeaf16826a9c72d1ba8166c1e3fab9b"],
            layer_256=["2ffef1834d5fe14ad8db58fc78d769d5dc38dda5eddbfc396786f74b326215fd"],
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
                    "diffusers": "Wav2Vec2ConformerForCTC",
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
