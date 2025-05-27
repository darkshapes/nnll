### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""神经网络的数据注册"""

# pylint: disable=possibly-used-before-assignment, line-too-long
from typing import Any, Callable, Union
import os

from nnll_01 import debug_monitor, nfo  # , dbug
from nnll_60 import MIR_PATH, JSONCache
from nnll_07 import mir_entry


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: dict
    mir_file = JSONCache(MIR_PATH)

    def __init__(self) -> None:
        self.read_from_disk()

    @debug_monitor
    def add(self, resource: dict[str, Any]) -> None:
        """Merge pre-existing MIR entries, or add new ones
        :param element: _description_
        """
        parent_key = next(iter(resource))
        if self.database.get(parent_key, 0):
            self.database[parent_key] = {**self.database[parent_key], **resource[parent_key]}
        else:
            self.database[parent_key] = resource[parent_key]

    @mir_file.decorator
    def write_to_disk(self, data: dict = None) -> None:  # pylint:disable=unused-argument
        """Save data to JSON file\n"""
        # from pprint import pprint

        self.mir_file.update_cache(self.database, replace=True)
        self.database = self.read_from_disk()
        nfo(self.database)
        nfo(f"Wrote {len(self.database)} lines to MIR database file.")

    @mir_file.decorator
    def read_from_disk(self, data: dict = None) -> dict:
        """Populate mir database\n
        :param data: mir decorater auto-populated, defaults to None
        :return: dict of MIR data"""
        self.database = data
        return self.database

    @debug_monitor
    def find_path(self, key: str, query: str, join_tag: bool = False) -> Any:
        """Retrieve MIR path based on nested value search\n
        :param key: Known field to look within
        :param query: Search pattern for field
        :param join_tag: Combine tag elements, defaults to False
        :return: A list or string of the found tag"""
        for series, comp in self.database.items():
            for compatibility, entry in comp.items():
                parameter = entry.get(key)
                nfo(f" maid found path {parameter} {key} {series} {compatibility} ")
                if parameter is not None and query.lower() in parameter:
                    found = "".join([series, compatibility]) if join_tag else [series, compatibility]
                    nfo(f" maid found path {parameter} {found} {series} {compatibility} ")
                    return found

    @staticmethod
    def grade_char_match(target: str, options: Union[list[str] | dict[str:Any]]) -> str | None:
        """Compare text to a sequence of texts and pick the closest match between them\n
        :param target: The ideal text
        :param options: The possible text matches
        :return: The closest match as a string, or `None`
        """

        closest_match = None
        min_difference = float("inf")
        for idx, opt in enumerate(options):
            entry = opt.lower()
            target_lower = target.lower()
            if target_lower in entry:
                if idx == target:
                    closest_match = idx
                    break
                common_prefix_length = len(os.path.commonprefix([entry, target_lower]))
                difference = abs(len(entry) - len(target_lower)) + (len(entry) - common_prefix_length)
                if difference < min_difference:
                    min_difference = difference
                    closest_match = idx
        return closest_match, entry


def build_mir_unet(mir_db: MIRDatabase):
    """Create mir unet info database"""
    # from nnll_01 import nfo
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="base",
            gen_kwargs={
                "num_inference_steps": 40,
                "denoising_end": 0.8,
                "output_type": "latent",
                "safety_checker": False,
            },
            init_kwargs={
                "use_safetensors": True,
            },
            dep_pkg={"diffusers": ["StableDiffusionXLPipeline"]},
            layer_256=["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
            weight_map="weight_maps/model.unet.stable-diffusion-xl:base.json",
            repo=["stabilityai/stable-diffusion-xl-base-1.0"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="refiner",
            repo=["stabilityai/stable-diffusion-xl-refiner-1.0"],
            layer_256=["8c2d0d32cff5a74786480bbaa932ee504bb140f97efdd1a3815f14a610cf6e4a"],
            weight_map="weight_maps/stable-diffusion-xl-refiner.json",
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            gen_kwargs={"num_inference_steps": 40, "denoising_end": 0.8},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="Kolors",
            comp="diffusers",
            repo=["Kwai-Kolors/Kolors-diffusers"],
            gen_kwargs={"negative_prompt": "", "guidance_scale": 5.0, "num_inference_steps": 50},
            init_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
            dep_pkg={"diffusers": ["KolorsPipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="combined",
            repo=["stabilityai/stable-cascade"],
            dep_pkg={"diffusers": ["StableCascadeCombinedPipeline"]},
            gen_kwargs={"negative_prompt": "", "num_inference_steps": 10, "prior_num_inference_steps": 20, "prior_guidance_scale": 3.0, "width": 1024, "height": 1024},
            init_kwargs={"variant": "bf16", "torch_dtype": "torch.bfloat16"},
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="prior",
            repo=["stabilityai/stable-cascade-prior"],
            dep_alt={"diffusers": ["StableCascadePriorPipeline"]},
            layer_256=[
                "2b6986954d9d2b0c702911504f78f5021843bd7050bb10444d70fa915cb495ea",
                "2aa5a461c4cd0e2079e81554081854a2fa01f9b876d7124c8fff9bf1308b9df7",
                "ce474fd5da12f1d465a9d236d61ea7e98458c1b9d58d35bb8412b2acb9594f08",
                "1b035ba92da6bec0a9542219d12376c0164f214f222955024c884e1ab08ec611",
                "22a49dc9d213d5caf712fbf755f30328bc2f4cbdc322bcef26dfcee82f02f147",
            ],
            init_kwargs={"variant": "bf16", "torch_dtype": "torch.bfloat16"},
            gen_kwargs={"height": 1024, "width": 1024, "negative_prompt": "", "guidance_scale": 4.0, "num_images_per_prompt": 1, "num_inference_steps": 20},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="decoder",
            repo=["stabilityai/stable-cascade"],
            dep_apt={"diffusers": ["StableCascadeDecoderPipeline"]},
            layer_256=[
                "fde5a91a908e8cb969f97bcd20e852fb028cc039a19633b0e1559ae41edeb16f",
                "24fa8b55d12bf904878b7f2cda47c04c1a92da702fe149e28341686c080dfd4f",
                "a7c96afb54e60386b7d077bf3f00d04596f4b877d58e6a577f0e1a08dc4a0190",
                "f1300b9ffe051640555bfeee245813e440076ef90b669332a7f9fb35fffb93e8",
                "047fa405c9cd5ad054d8f8c8baa2294fbc663e4121828b22cb190f7057842a64",
            ],
            init_kwargs={"variant": "bf16", "torch_dtype": "torch.bfloat16"},
            gen_kwargs={"guidance_scale": 0.0, "output_type": "pil", "num_inference_steps": 10},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="pony-diffusion",
            layer_256=["d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
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
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2.5-base",
            layer_256=["a6f31493ceeb51c88c5239188b9078dc64ba66d3fc5958ad48c119115b06120c"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2.5-aesthetic",
            repo=["playgroundai/playground-v2.5-1024px-aesthetic"],
            layer_256=[
                "fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20",
                "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843",
            ],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            init_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
            gen_kwargs={"num_inference_steps": 50, "guidance_scale": 3},
        )
    )


def build_mir_dit(mir_db: MIRDatabase):
    """Create mir diffusion transformer info database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="medium",
            repo=["stabilityai/stable-diffusion-3.5-medium", "adamo1139/stable-diffusion-3.5-medium-ungated"],
            layer_256=["dee29a467c44cff413fcf1c2dda0b31f5f0a4e093029a8e5a05140f40ae061ee"],
            dep_pkg={"diffusers": ["StableDiffusion3Pipeline"]},
            gen_kwargs={"num_inference_steps": 40, "guidance_scale": 4.5},
            init_kwargs={"torch_dtype": "torch.float16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="large",
            repo=["stabilityai/stable-diffusion-3.5-large", "adamo1139/stable-diffusion-3.5-large-ungated"],
            layer_256=["8c2e5bc99bc89290254142469411db66cb2ca2b89b129cd2f6982b30e26bd465"],
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5},
            init_kwargs={"torch_dtype": "torch.float16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="gguf",
            layer_256=["e7eddc3cd09ccf7c9c03ceef70bbcd91d44d46673857d37c3abfe4e6ee240a96"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="alchemist-large",
            repo=["yandex/stable-diffusion-3.5-large-alchemist"],
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="alchemist-medium",
            repo=["yandex/stable-diffusion-3.5-medium-alchemist"],
            gen_kwargs={"num_inference_steps": 40, "guidance_scale": 4.5},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="schnell",
            repo=["black-forest-labs/flux.1-schnell", "cocktailpeanut/xulf-s"],
            layer_256=["ef5c9cd1ebe6e3be5e8b1347eca0a6f0b138986c71220a7f1c2c14f29d01beed"],
            dep_pkg={"diffusers": ["FluxPipeline"]},
            gen_kwargs={"guidance_scale": 0.0, "num_inference_steps": 4, "max_sequence_length": 256},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="shuttle-3.1-aesthetic",
            repo=["shuttleai/shuttle-3.1-aesthetic"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 4, "max_sequence_length": 256},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="shuttle-3-diffusion",
            repo=["shuttleai/shuttle-3-diffusion"],
            dep_alt={"diffusers": ["DiffusionPipeline"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 4, "max_sequence_length": 256},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="dev",
            repo=["black-forest-labs/flux.1-dev", "cocktailpeanut/xulf-d"],
            layer_256=[
                "ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881",
                "20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9",
            ],
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 50, "max_sequence_length": 512},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="f-lite-8b",
            repo=["freepik/flux.1-lite-8b"],
            dep_repo=["github.com/fal-ai/f-lite.git"],
            dep_alt={"f_lite": ["FLitePipeline"]},
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5, "height": 1024, "width": 1024},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="f-lite-7b",
            repo=["freepik/f-lite-7b"],
            dep_repo=["github.com/fal-ai/f-lite.git"],
            dep_alt={"f_lite": ["FLitePipeline"]},
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5, "height": 1024, "width": 1024},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="lite-texture",
            repo=["freepik/f-lite-texture"],
            dep_repo=["github.com/fal-ai/f-lite.git"],
            dep_alt={"f_lite": ["FLitePipeline"]},
            gen_kwargs={"num_inference_steps": 28, "guidance_scale": 3.5, "height": 1024, "width": 1024},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
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
            series="flux-1",
            comp="mini",
            repo=["TencentARC/flux-mini"],
            dep_alt={"diffusers": ["diffusers"]},
            dep_repo=["TencentARC/FluxKits"],
            layer_256=["e4a0d8cf2034da094518ab058da1d4aea14e00d132c6152a266ec196ffef02d0"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="auraflow",
            comp="0",
            dep_pkg={
                "diffusers": ["AuraFlowPipeline"],
            },
            repo=["fal/AuraFlow-v0.3", "fal/AuraFlow-v0.2", "fal/AuraFlow"],
            gen_kwargs={"width": 1536, "height": 768, "num_inference_steps": 50, "guidance_scale": 3.5},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="hunyuandit",
            comp="diffusers",
            dep_pkg={"diffusers": ["HunyuanDiTPipeline"]},
            repo=["Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"],
            gen_kwargs={"num_inference_steps": 50, "guidance_scale": 6},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="hunyuandit",
            comp="distilled",
            repo=["Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"],
            gen_kwargs={"num_inference_steps": 25},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="pixart-sigma",
            comp="xl-2-1024",
            dep_pkg={"diffusers": ["PixArtSigmaPipeline"]},
            repo=["PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"],
            init_kwargs={"torch_dtype": "torch.float16", "use_safetensors": True},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="cogview-3",
            comp="plus-3b",
            repo=["THUDM/CogView3-Plus-3B"],
            dep_pkg={"diffusers": ["CogView3PlusPipeline"]},
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 4.0, "num_inference_steps": 50},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="cogview-4",
            comp="6b",
            repo=["THUDM/CogView4-6B"],
            dep_pkg={"diffusers": ["CogView4Pipeline"]},
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_images_per_prompt": 1, "num_inference_steps": 50},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="lumina-image",
            comp="2.0",
            repo=["Alpha-VLLM/Lumina-Image-2.0"],
            gen_kwargs={
                "height": 1024,
                "width": 1024,
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "cfg_trunc_ratio": 0.25,
                "cfg_normalization": True,
            },
            dep_pkg={"diffusers": ["Lumina2Pipeline"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="fuse-dit",
            comp="2b",
            repo=["ooutlierr/fuse-dit"],
            dep_repo=["github.com/tang-bd/fuse-dit.git"],
            dep_pkg={"diffusion": ["pipelines.FuseDiTPipeline"]},
            gen_kwargs={
                "width": 512,
                "height": 512,
                "num_inference_steps": 25,
                "guidance_scale": 6.0,
                "use_cache": True,
            },
        )
    )


def build_mir_art(mir_db: MIRDatabase):
    """Create mir autoregressive info database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="phi-4",
            comp="multimodal-instruct",
            repo=["microsoft/Phi-4-multimodal-instruct"],
            dep_pkg={"transformers": ["AutoModelForCausalLM"]},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="audiogen",
            comp="medium-1.5b",
            repo=["facebook/audiogen-medium"],
            dep_pkg={"audiocraft": ["models", "AudioGen"]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="parler-tts",
            comp="tiny-v1",
            repo=["parler-tts/parler-tts-tiny-v1"],
            dep_pkg={"parler_tts": ["ParlerTTSForConditionalGeneration"], "transformers": ["AutoTokenizer"]},
            init_kwargs={"AutoTokenizer": {"return_tensors": "pt"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="parler-tts",
            comp="large-v1",
            repo=["parler-tts/parler-tts-large-v1"],
            dep_pkg={"parler_tts": ["ParlerTTSForConditionalGeneration"], "transformers": ["AutoTokenizer"]},
            init_kwargs={"AutoTokenizer": {"return_tensors": "pt"}},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="lumina-mgpt",
            comp="7B-768",
            repo=["Alpha-VLLM/Lumina-mGPT-7B-768"],
            dep_repo=["github.com/Alpha-VLLM/Lumina-mGPT"],
            dep_pkg={"inference_solver": ["FlexARInferenceSolver"]},
            init_kwargs={"precision": "bf16", "target_size": 768},
            gen_kwargs={"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
        )
    )


def build_mir_mix(mir_db: MIRDatabase):
    """mixed-type architecture"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="mix",
            series="bagel",
            comp="7B-MoT",
            repo="ByteDance-Seed/BAGEL-7B-MoT",
            dep_repo=["github.com/ByteDance-Seed/Bagel/"],
        )
    )


def build_mir_lora(mir_db: MIRDatabase):
    """Create mir lora database"""
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dmd",
            comp="stable-diffusion-xl",
            repo=["tianweiy/DMD2/"],
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={},
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0, "timesteps": [999, 749, 499, 249]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dpo",
            comp="stable-diffusion-xl",
            repo=["radames/sdxl-DPO-LoRA"],
            scheduler="ops.scheduler.dpm",
            scheduler_kwargs={"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "order": 2},
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"guidance_scale": 7.5, "num_inference_steps": 4},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-xl",
            repo=["jasperai/flash-sdxl"],
            scheduler="ops.scheduler.lcm",
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler_kwargs={},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="flash", comp="pixart-alpha", repo=["jasperai/flash-pixart"]))
    mir_db.add(mir_entry(domain="info", arch="lora", series="flash", comp="stable-diffusion-3", repo=["jasperai/flash-sd3"]))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-1",
            repo=["jasperai/flash-sd"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            dep_pkg={"diffusers": ["diffusers"]},
            comp="stable-diffusion-xl",
            repo=["ByteDance/Hyper-SD"],
            init_kwargs={"fuse": 1.0},
        ),
    )
    mir_db.add(
        mir_entry(domain="info", arch="lora", series="hyper", comp="flux-1:dev", repo=["ByteDance/Hyper-SD"], init_kwargs={"fuse": 0.125}),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp="stable-diffusion-3",
            repo=["ByteDance/Hyper-SD"],
            init_kwargs={"fuse": 0.125},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="hyper", comp="stable-diffusion-1", repo=["ByteDance/Hyper-SD"]))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="stable-diffusion-xl",
            repo=["latent-consistency/lcm-lora-sdxl"],
            init_kwargs={"fuse": 1.0},
            gen_kwargs={
                "num_inference_steps": 8,
            },
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={"timestep_spacing": "trailing"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="ssd-1b",
            repo=["latent-consistency/lcm-lora-ssd-1b"],
            gen_kwargs={"num_inference_steps": 8},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="vega",
            repo=["segmind/Segmind-VegaRT"],
            gen_kwargs={"num_inference_steps": 8},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="stable-diffusion-1",
            repo=["latent-consistency/lcm-lora-sdv1-5"],
            gen_kwargs={"num_inference_steps": 8},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lightning",
            comp="stable-diffusion-xl",
            repo=["ByteDance/SDXL-Lightning"],
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="pcm", comp="stable-diffusion-xl", repo=["wangfuyun/PCM_Weights"]))
    mir_db.add(mir_entry(domain="info", arch="lora", series="pcm", comp="stable-diffusion-1", repo=["wangfuyun/PCM_Weights"]))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="slam",
            comp="stable-diffusion-xl",
            repo=["alimama-creative/slam-lora-sdxl/"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 1},
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={"timestep_spacing": "trailing"},
        )
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="slam", comp="stable-diffusion-1", repo=["alimama-creative/slam-sd1.5"]))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-xl",
            repo=["SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA"],
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"guidance_scale": 5.0},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-1",
            repo=["SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA"],
            gen_kwargs={"guidance_scale": 7.5},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp="stable-diffusion-xl",
            repo=["h1t/TCD-SDXL-LoRA"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0, "eta": 0.3},
            dep_pkg={"diffusers": ["diffusers"]},
            scheduler="ops.scheduler.tcd",
            scheduler_kwargs={},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="tcd", comp="stable-diffusion-1", repo=["h1t/TCD-SD15-LoRA"]))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp="flux-1:dev",
            repo=["alimama-creative/FLUX.1-Turbo-Alpha"],
            dep_pkg={"diffusers": ["diffusers"]},
            gen_kwargs={"guidance_scale": 3.5, "num_inference_steps": 8, "max_sequence_length": 512},
            init_kwargs={"fuse": 0.125},
        )
    )


def build_mir_other(mir_db: MIRDatabase):
    """Create mir info database"""
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dep_pkg={"hidiffusion": ["apply_hidiffusion"]},
            repo=["github.com/megvii-research/HiDiffusion/"],
            gen_kwargs={"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5},
        )
    )


def build_mir_float(mir_db: MIRDatabase):
    """Create mir info database"""
    mir_db.add(mir_entry(domain="ops", arch="float", series="BF16", comp="pytorch", dep_pkg={"torch": ["bfloat16"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="F16", comp="pytorch", variant="fp16", dep_pkg={"torch": ["float16"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="F32", comp="pytorch", variant="fp32", dep_pkg={"torch": ["float32"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="F64", comp="pytorch", variant="fp64", dep_pkg={"torch": ["float64"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="F8_E4M3", comp="pytorch", variant="fp8e4m3fn", dep_pkg={"torch": ["float8_e4m3fn"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="F8_E5M2", comp="pytorch", variant="fp8e5m2", dep_pkg={"torch": ["float8_e5m2"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="I16", comp="pytorch", dep_pkg={"torch": ["int16"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="I32", comp="pytorch", dep_pkg={"torch": ["int32"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="I64", comp="pytorch", dep_pkg={"torch": ["int64"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="I8", comp="pytorch", dep_pkg={"torch": ["int8"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="NF4", comp="pytorch", dep_pkg={"torch": ["nf4"]}))
    mir_db.add(mir_entry(domain="ops", arch="float", series="U8", comp="pytorch", dep_pkg={"torch": ["uint8"]}))


def build_mir_scheduler(mir_db: MIRDatabase):
    """Create mir info database"""
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="euler", comp="[init]", dep_pkg={"diffusers": ["EulerDiscreteScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="euler-ancestral", comp="[init]", dep_pkg={"diffusers": ["EulerAncestralDiscreteScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="flow-match", comp="[init]", dep_pkg={"diffusers": ["FlowMatchEulerDiscreteScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="edm", comp="[init]", dep_pkg={"diffusers": ["EDMDPMSolverMultistepScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="dpm", comp="[init]", dep_pkg={"diffusers": ["DPMSolverMultistepScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="ddim", comp="[init]", dep_pkg={"diffusers": ["DDIMScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="lcm", comp="[init]", dep_pkg={"diffusers": ["LCMScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="tcd", comp="[init]", dep_pkg={"diffusers": ["TCDScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="heun", comp="[init]", dep_pkg={"diffusers": ["HeunDiscreteScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="uni-pc", comp="[init]", dep_pkg={"diffusers": ["UniPCMultistepScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="lms", comp="[init]", dep_pkg={"diffusers": ["LMSDiscreteScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="deis", comp="[init]", dep_pkg={"diffusers": ["DEISMultistepScheduler"]}))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="ddpm_wuerstchen", comp="[init]", dep_pkg={"diffusers": ["DDPMWuerstchenScheduler"]}))
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dep_alt={"diffusers": ["schedulers.scheduling_utils", "AysSchedules"]},
        )
    )


def main(mir_db: Callable = MIRDatabase()) -> None:
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # root_path = os.path.dirname(current_dir)
    # sys.path.append(root_path)
    # sys.path.append(os.getcwd())
    # print(sys.path)

    build_mir_unet(mir_db)
    build_mir_dit(mir_db)
    build_mir_art(mir_db)
    build_mir_lora(mir_db)
    build_mir_scheduler(mir_db)
    build_mir_float(mir_db)
    build_mir_other(mir_db)
    mir_db.write_to_disk()


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    main()
