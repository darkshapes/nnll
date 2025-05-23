### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from typing import Any
from nnll_60 import MIR_PATH, JSONCache
from nnll_07 import mir_entry


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: dict
    mir_file = JSONCache(MIR_PATH)

    def __init__(self) -> None:
        self.database = {}

    def add(self, resource: dict[str, Any]) -> None:
        """Merge pre-existing MIR entries, or add new ones
        :param element: _description_
        """
        parent_key = next(iter(resource))
        if self.database.get(parent_key, 0):
            self.database[parent_key] = {**self.database[parent_key], **resource[parent_key]}
        else:
            self.database[parent_key] = resource[parent_key]

    def write_to_disk(self) -> None:
        """Save data to JSON file\n"""
        self.mir_file.update_cache(self.database)

    @mir_file.decorator
    def read_from_disk(self, data: dict = None) -> dict:
        self.database = data


mir_db = MIRDatabase()


def build_mir_unet():
    """Create mir info database"""
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
            dep_pkg=["diffusers", "transformers"],
            layer_256=["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
            weight_map="weight_maps/model.unet.stable-diffusion-xl:base.json",
            repo="stabilityai/stable-diffusion-xl-base-1.0",
            module_path=["StableDiffusionXLPipeline"],
            module_alt=["DiffusionPipeline"],
            module_i2i=["StableDiffusionXLImg2ImgPipeline"],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="refiner",
            repo="stabilityai/stable-diffusion-xl-refiner-1.0",
            layer_256=["8c2d0d32cff5a74786480bbaa932ee504bb140f97efdd1a3815f14a610cf6e4a"],
            weight_map="weight_maps/stable-diffusion-xl-refiner.json",
            module_alt=["DiffusionPipeline"],
            gen_kwargs={"num_inference_steps": 40, "denoising_end": 0.8},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="Kolors",
            comp="diffusers",
            repo="Kwai-Kolors/Kolors-diffusers",
            gen_kwargs={"negative_prompt": "", "guidance_scale": 5.0, "num_inference_steps": 50},
            init_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
            module_path=["KolorsPipeline"],
            dep_pkg=["diffusers", "transformers"],
        )
    )

    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            dep_pkg=["diffusers", "transformers"],
            comp="c",
            layer_256=[
                "2b6986954d9d2b0c702911504f78f5021843bd7050bb10444d70fa915cb495ea",
                "2aa5a461c4cd0e2079e81554081854a2fa01f9b876d7124c8fff9bf1308b9df7",
                "ce474fd5da12f1d465a9d236d61ea7e98458c1b9d58d35bb8412b2acb9594f08",
                "1b035ba92da6bec0a9542219d12376c0164f214f222955024c884e1ab08ec611",
                "22a49dc9d213d5caf712fbf755f30328bc2f4cbdc322bcef26dfcee82f02f147",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
            comp="b",
            layer_256=[
                "fde5a91a908e8cb969f97bcd20e852fb028cc039a19633b0e1559ae41edeb16f",
                "24fa8b55d12bf904878b7f2cda47c04c1a92da702fe149e28341686c080dfd4f",
                "a7c96afb54e60386b7d077bf3f00d04596f4b877d58e6a577f0e1a08dc4a0190",
                "f1300b9ffe051640555bfeee245813e440076ef90b669332a7f9fb35fffb93e8",
                "047fa405c9cd5ad054d8f8c8baa2294fbc663e4121828b22cb190f7057842a64",
            ],
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="pony-diffusion",
            layer_256=["d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16"],
            module_path=["StableDiffusionXLPipeline"],
            module_alt=["DiffusionPipeline"],
            dep_pkg=["diffusers", "transformers"],
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
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-diffusion-xl",
            comp="playground-2.5-aesthetic",
            repo="playgroundai/playground-v2.5-1024px-aesthetic",
            layer_256=[
                "fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20",
                "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843",
            ],
            module_alt="DiffusionPipeline",
            init_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
            gen_kwargs={"num_inference_steps": 50, "guidance_scale": 3},
        )
    )


def build_mir_dit():
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="medium",
            repo="stabilityai/stable-diffusion-3.5-medium",
            layer_256=["dee29a467c44cff413fcf1c2dda0b31f5f0a4e093029a8e5a05140f40ae061ee"],
            repo_alt=["adamo1139/stable-diffusion-3.5-medium-ungated"],
            dep_pkg=["diffusers", "transformers"],
            module_path=["StableDiffusion3Pipeline"],
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
            repo="stabilityai/stable-diffusion-3.5-large",
            layer_256=["8c2e5bc99bc89290254142469411db66cb2ca2b89b129cd2f6982b30e26bd465"],
            repo_alt=["adamo1139/stable-diffusion-3.5-large-ungated"],
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
            repo="yandex/stable-diffusion-3.5-large-alchemist",
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
            repo="yandex/stable-diffusion-3.5-medium-alchemist",
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
            repo="black-forest-labs/flux.1-schnell",
            layer_256=["ef5c9cd1ebe6e3be5e8b1347eca0a6f0b138986c71220a7f1c2c14f29d01beed"],
            module_path=["FluxPipeline"],
            repo_alt=["cocktailpeanut/xulf-s"],
            dep_pkg=["diffusers", "transformers"],
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
            repo="shuttleai/shuttle-3.1-aesthetic",
            module_alt=["DiffusionPipeline"],
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
            repo="shuttleai/shuttle-3-diffusion",
            module_alt=["DiffusionPipeline"],
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
            repo="black-forest-labs/flux.1-dev",
            layer_256=[
                "ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881",
                "20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9",
            ],
            repo_alt=["cocktailpeanut/xulf-d"],
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 50, "max_sequence_length": 512},
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="flux-1",
            comp="lite",
            repo="Freepik/flux.1-lite-8B",
            repo_alt=["Freepik/F-Lite-Texture"],
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
            dep_pkg=["diffusers", "transformers"],
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
            dep_pkg=["diffusers", "transformers"],
            layer_256=["e4a0d8cf2034da094518ab058da1d4aea14e00d132c6152a266ec196ffef02d0"],
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="auraflow",
            comp="0",
            dep_pkg=["diffusers", "transformers"],
            repo="fal/AuraFlow-v0.3",
            repo_alt=["fal/AuraFlow-v0.2", "fal/AuraFlow"],
            module_path=["AuraFlowPipeline"],
            gen_kwargs={"width": 1536, "height": 768, "num_inference_steps": 50, "guidance_scale": 3.5},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="hunyuandit",
            comp="diffusers",
            dep_pkg=["diffusers", "transformers"],
            repo="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
            module_path=["HunyuanDiTPipeline"],
            gen_kwargs={"num_inference_steps": 50, "guidance_scale": 6},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="hunyuandit",
            comp="distilled",
            repo="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
            gen_kwargs={"num_inference_steps": 25},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="pixart-sigma",
            comp="xl-2-1024",
            dep_pkg=["diffusers", "transformers"],
            repo="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            module_path=["PixArtSigmaPipeline"],
            init_kwargs={"torch_dtype": "torch.float16", "use_safetensors": True},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="cogview-3",
            comp="plus-3b",
            repo="THUDM/CogView3-Plus-3B",
            dep_pkg=["diffusers", "transformers"],
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 4.0, "num_inference_steps": 50},
            module_path=["CogView3PlusPipeline"],
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="cogview-4",
            comp="6b",
            repo="THUDM/CogView4-6B",
            dep_pkg=["diffusers", "transformers"],
            gen_kwargs={"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_images_per_prompt": 1, "num_inference_steps": 50},
            module_path=["CogView4Pipeline"],
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="lumina-image",
            comp="2.0",
            repo="Alpha-VLLM/Lumina-Image-2.0",
            gen_kwargs={
                "height": 1024,
                "width": 1024,
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "cfg_trunc_ratio": 0.25,
                "cfg_normalization": True,
            },
            dep_pkg=["diffusers", "transformers"],
            module_path=["Lumina2Pipeline"],
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )


def build_mir_art():
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="phi-4",
            comp="multimodal-instruct",
            repo="microsoft/Phi-4-multimodal-instruct",
            module_path=["AutoModelForCausalLM"],
            dep_pkg=["transformers"],
            init_kwargs={"torch_dtype": "torch.bfloat16"},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="art",
            series="lumina-mgpt",
            comp="7B-768",
            repo="Alpha-VLLM/Lumina-mGPT-7B-768",
            dep_pkg=["github.com/Alpha-VLLM/Lumina-mGPT"],
            module_path=["FlexARInferenceSolver"],
            init_kwargs={"precision": "bf16", "target_size": 768},
            gen_kwargs={"images": [], "qas": [["q1", None]], "max_gen_len": 8192, "temperature": 1.0},
        )
    )


def build_mir_lora():
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dmd",
            comp="stable-diffusion-xl",
            repo="tianweiy/DMD2/",
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={},
            dep_pkg=["diffusers"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0, "timesteps": [999, 749, 499, 249]},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="dpo",
            comp="stable-diffusion-xl",
            repo="radames/sdxl-DPO-LoRA",
            scheduler="ops.scheduler.dpm",
            scheduler_kwargs={"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True, "order": 2},
            dep_pkg=["diffusers"],
            gen_kwargs={"guidance_scale": 7.5, "num_inference_steps": 4},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-xl",
            repo="jasperai/flash-sdxl",
            scheduler="ops.scheduler.lcm",
            dep_pkg=["diffusers"],
            scheduler_kwargs={},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="flash", comp="pixart-alpha", repo="jasperai/flash-pixart"))
    mir_db.add(mir_entry(domain="info", arch="lora", series="flash", comp="stable-diffusion-3", repo="jasperai/flash-sd3"))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="flash",
            comp="stable-diffusion-1",
            repo="jasperai/flash-sd",
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            dep_pkg=["diffusers"],
            comp="stable-diffusion-xl",
            repo="ByteDance/Hyper-SD",
            init_kwargs={"fuse": 1.0},
        ),
    )
    mir_db.add(
        mir_entry(domain="info", arch="lora", series="hyper", comp="flux-1:dev", repo="ByteDance/Hyper-SD", init_kwargs={"fuse": 0.125}),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="hyper",
            comp="stable-diffusion-3",
            repo="ByteDance/Hyper-SD",
            init_kwargs={"fuse": 0.125},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="hyper", comp="stable-diffusion-1", repo="ByteDance/Hyper-SD"))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="stable-diffusion-xl",
            repo="latent-consistency/lcm-lora-sdxl",
            init_kwargs={"fuse": 1.0},
            gen_kwargs={
                "num_inference_steps": 8,
            },
            dep_pkg=["diffusers"],
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
            repo="latent-consistency/lcm-lora-ssd-1b",
            gen_kwargs={"num_inference_steps": 8},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="vega",
            repo="segmind/Segmind-VegaRT",
            gen_kwargs={"num_inference_steps": 8},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lcm",
            comp="stable-diffusion-1",
            repo="latent-consistency/lcm-lora-sdv1-5",
            gen_kwargs={"num_inference_steps": 8},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="lightning",
            comp="stable-diffusion-xl",
            repo="ByteDance/SDXL-Lightning",
            dep_pkg=["diffusers"],
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="pcm", comp="stable-diffusion-xl", repo="wangfuyun/PCM_Weights"))
    mir_db.add(mir_entry(domain="info", arch="lora", series="pcm", comp="stable-diffusion-1", repo="wangfuyun/PCM_Weights"))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="slam",
            comp="stable-diffusion-xl",
            repo="alimama-creative/slam-lora-sdxl/",
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 1},
            dep_pkg=["diffusers"],
            scheduler="ops.scheduler.lcm",
            scheduler_kwargs={"timestep_spacing": "trailing"},
        )
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="slam", comp="stable-diffusion-1", repo="alimama-creative/slam-sd1.5"))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-xl",
            repo="SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA",
            dep_pkg=["diffusers"],
            gen_kwargs={"guidance_scale": 5.0},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="spo",
            comp="stable-diffusion-1",
            repo="SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA",
            gen_kwargs={"guidance_scale": 7.5},
        ),
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="tcd",
            comp="stable-diffusion-xl",
            repo="h1t/TCD-SDXL-LoRA",
            gen_kwargs={"num_inference_steps": 4, "guidance_scale": 0, "eta": 0.3},
            dep_pkg=["diffusers"],
            scheduler="ops.scheduler.tcd",
            scheduler_kwargs={},
        ),
    )
    mir_db.add(mir_entry(domain="info", arch="lora", series="tcd", comp="stable-diffusion-1", repo="h1t/TCD-SD15-LoRA"))
    mir_db.add(
        mir_entry(
            domain="info",
            arch="lora",
            series="turbo",
            comp="flux-1:dev",
            repo="alimama-creative/FLUX.1-Turbo-Alpha",
            dep_pkg=["diffusers"],
            gen_kwargs={"guidance_scale": 3.5, "num_inference_steps": 8, "max_sequence_length": 512},
            init_kwargs={"fuse": 0.125},
        )
    )


def build_mir_other():
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dep_pkg=["hidiffusion"],
            repo="megvii-research/HiDiffusion/",
            gen_kwargs={"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5},
            module_path=["apply_hidiffusion"],
        )
    )


def build_mir_float():
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="BF16", dtype="torch.bfloat16"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="F16", variant="fp16", dtype="torch.float16"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="F32", variant="fp32", dtype="torch.float32"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="F64", variant="fp64", dtype="torch.float64", dep_pkg=["torch"]))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="F8_E4M3", variant="fp8e4m3fn", dtype="torch.float8_e4m3fn"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="F8_E5M2", variant="fp8e5m2", dtype="torch.float8_e5m2"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="I16", dtype="torch.int16"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="I32", dtype="torch.int32"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="I64", dtype="torch.int64"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="I8", dtype="torch.int8"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="NF4", dtype="nf4"))
    mir_db.add(mir_entry(domain="ops", arch="float", series="pytorch", comp="U8", dtype="torch.uint8"))


def build_mir_scheduler():
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="euler", comp="any", deps_pkg=["diffusers"], module_path=["EulerDiscreteScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="euler-ancestral", comp="any", deps_pkg=["diffusers"], module_path=["EulerAncestralDiscreteScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="flow-match", comp="any", deps_pkg=["diffusers"], module_path=["FlowMatchEulerDiscreteScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="edm", comp="any", deps_pkg=["diffusers"], module_path=["EDMDPMSolverMultistepScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="dpm", comp="any", deps_pkg=["diffusers"], module_path=["DPMSolverMultistepScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="ddim", comp="any", deps_pkg=["diffusers"], module_path=["DDIMScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="lcm", comp="any", deps_pkg=["diffusers"], module_path=["LCMScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="tcd", comp="any", deps_pkg=["diffusers"], module_path=["TCDScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="heun", comp="any", deps_pkg=["diffusers"], module_path=["HeunDiscreteScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="uni-pc", comp="any", deps_pkg=["diffusers"], module_path=["UniPCMultistepScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="lms", comp="any", deps_pkg=["diffusers"], module_path=["LMSDiscreteScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="deis", comp="any", deps_pkg=["diffusers"], module_path=["DEISMultistepScheduler"]))
    mir_db.add(mir_entry(domain="ops", arch="scheduler", series="ddpm_wuerstchen", comp="any", deps_pkg=["diffusers"], module_path=["DDPMWuerstchenScheduler"]))
    mir_db.add(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dep_pkg=["diffusers"],
            module_path=["schedulers.scheduling_utils", "AysSchedules"],
        )
    )


if __name__ == "__main__":
    from pprint import pprint

    build_mir_unet()
    build_mir_dit()
    build_mir_art()
    build_mir_lora()
    build_mir_scheduler()
    build_mir_float()
    build_mir_other()
    mir_db.write_to_disk()
    pprint(mir_db.database)
