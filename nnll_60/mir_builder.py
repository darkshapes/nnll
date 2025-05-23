### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from typing import Any


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: dict

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


def build_mir_db():
    """Create mir info database"""
    # from nnll_01 import nfo
    from nnll_07 import mir_entry
    from pprint import pprint

    mir_db = MIRDatabase()
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
            dep_pkg=["diffusers"],
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
            dep_pkg=["diffusers"],
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
            dep_pkg=["diffusers"],
        )
    )
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
    mir_db.add(
        mir_entry(
            domain="info",
            arch="unet",
            series="stable-cascade",
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
            dep_pkg=["diffusers"],
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
            layer_256=["fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20", "d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843"],
            module_alt="DiffusionPipeline",
            pipe_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
            gen_kwargs={"num_inference_steps": 50, "guidance_scale": 3},
        )
    )
    mir_db.add(
        mir_entry(
            domain="info",
            arch="dit",
            series="stable-diffusion-3",
            comp="medium",
            repo="stabilityai/stable-diffusion-3.5-medium",
            layer_256=["dee29a467c44cff413fcf1c2dda0b31f5f0a4e093029a8e5a05140f40ae061ee"],
            repo_alt=["adamo1139/stable-diffusion-3.5-medium-ungated"],
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
        )
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
    pprint(mir_db.database)


if __name__ == "__main__":
    build_mir_db()
