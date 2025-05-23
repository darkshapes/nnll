### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def test_mir_creation():
    from nnll_01 import nfo
    from nnll_07 import add_mir_entry
    from pprint import pprint

    entry = add_mir_entry(
        domain="info", arch="unet", series="stable-diffusion-xl", compatibility="base", gen_kwargs={"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False}, pipe_kwargs={"use_safetensors": True}
    )
    entry.update(
        add_mir_entry(domain="model", arch="unet", series="stable-diffusion-xl", compatibility="base", file_path="/Users/nyan/Documents/models"),
    )
    entry.update(
        add_mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            compatibility="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dependency="diffusers",
            module_path=["schedulers.scheduling_utils", "AysSchedules"],
        )
    )
    entry.update(
        add_mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            compatibility="stable-diffusion-xl",
            num_inference_steps=10,
            timesteps="StableDiffusionXLTimesteps",
            dependency="hidiffusion",
            gen_kwargs={"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5},
            module_path=["apply_hidiffusion"],
        )
    )
    pprint(entry)


# eta only works with ddim!!!

#  ays_type="StableDiffusionXLTimesteps") -> Tuple[Callable, dict]:
#     """Apply AlignYourSteps optimization
#     compatibility: stable-diffusion-xl, stable-diffusion, stable-video-diffusion
#     """

# domain_info = Domain("model")
# arch_name = Architecture("unet")
# impl_sdxl = Implementation("stable-diffusion-xl")
# impl_sdxl.pipelines = "StableDiffusionXL"
# compat = Compatibility(
#     "base",
#     Model(file_path="/Users/nyan/Documents/models"),
# )
# impl_sdxl.add_compat(compat.stage, compat)
# arch_name.add_impl(impl_sdxl.implementation, impl_sdxl)
# domain_info.add_arch(arch_name.architecture, arch_name)
# label = domain_info.to_dict()
# print(label)


# domain_info = Domain("info")
# arch_name = Architecture("unet")
# impl_sdxl = Series("stable-diffusion-xl")
# compat = Compatibility(
#     "base",
#     defaults={"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
#     pipe_kwargs={"use_safetensors": True},
# )
# impl_sdxl.add_compat(compat.component_name, compat)
# arch_name.add_impl(impl_sdxl.series, impl_sdxl)
# domain_info.add_arch(arch_name.architecture, arch_name)
# label = domain_info.to_dict()
# print(label)

# dependencies: str  # importable dependency mapfor the series
# pipelines: Optional[str] = None  # name of sub package of import
# remote_paths: Optional[list[urllib.parse.urlparse]]


#     domain_info = Domain("info")
#     arch_name = Architecture("unet")
#     impl_sdxl = Implementation("stable-diffusion-xl")
#     compat = Compatibility(
#         "base",
#         metadata=Info(
#             gen_kwargs={"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
#             pipe_kwargs={"use_safetensors": True},
#         ),
#     )
#     impl_sdxl.add_compat(compat.stage, compat)
#     arch_name.add_impl(impl_sdxl.implementation, impl_sdxl)
#     domain_info.add_arch(arch_name.architecture, arch_name)
#     label = domain_info.to_dict()
#     print(label)

# domain_info = Domain("model")
# arch_name = Architecture("unet")
# impl_sdxl = Implementation("stable-diffusion-xl")
# impl_sdxl.pipelines = "StableDiffusionXL"
# compat = Compatibility(
#     "base",
#     Model(file_path="/Users/nyan/Documents/models"),
# )
# impl_sdxl.add_compat(compat.stage, compat)
# arch_name.add_impl(impl_sdxl.implementation, impl_sdxl)
# domain_info.add_arch(arch_name.architecture, arch_name)
# label = domain_info.to_dict()
# print(label)


if __name__ == "__main__":
    test_mir_creation()

# from nnll_07 import *
# info = Domain('info')
# arch_name = Architecture('unet')
# ver = Implementation('stable-diffusion-xl')
# ver.pipelines = "StableDiffusionXL"
# compat = Compatibility('base',defaults={"num_inference_steps": 40,"denoising_end": 0.8,"output_type": "latent","safety_checker": False}, pipe_kwargs={"use_safetensors": True})

# compat.to_dict()
# ver.add_compat(compat)
# ver.to_dict()
# arch_name.add_impl(ver)
# arch_name.to_dict()
# info.add_arch(arch_name)
# info.to_dict()

# `domain_name`  [ml/info/dev] see `domain class` for details
# `architecture` the family and version (stable diffusion 3.5, lumina next)
# `components`   attributes and process stage (lora, unet)

# # Create a domain
# `domain_ml = Domain("ml")`

# # Create architectures within the domain
# ```
# impl_sdxl = Implementation("stable-diffusion-xl")
# arch_auraflow = Architecture("auraflow")
# arch_flux = Architecture("flux")
# ```

# # Create components within architectures
# ```
# comp_xl = Compatibility("base", dtype="float32", file_size=1024, layer_type="diffusers")
# comp_vae = Component("vae", dtype="float32", file_size=512, layer_type="diffusers")
# comp_lora = Component("lora", dtype="float32", file_size=256, layer_type="diffusers")
# ```

# # Add components to architectures
# ```
# arch_sdxl_base.add_component(comp_unet.component_name, comp_unet)
# arch_auraflow.add_component(comp_vae.component_name, comp_vae)
# arch_flux.add_component(comp_lora.component_name, comp_lora)
# ```

# # Add architectures to domain
# domain_ml.add_architecture(arch_sdxl_base.architecture, arch_sdxl_base)
# domain_ml.add_architecture(arch_auraflow.architecture, arch_auraflow)
# domain_ml.add_architecture(arch_flux.architecture, arch_flux)

# # Serialize the domain to a dictionary for storage or transmission
# model_index_dict = domain_ml.to_dict()
# print(model_index_dict)

# import unittest

# import os
# import sys

# from nnll_07 import Domain, Architecture, Compatability, Implementation


# class TestDomainArchitectureCompatability(unittest.TestCase):
#     def test_domain_architecture_component(self):
#         # Create a Compatability object
#         component1 = Compatability(model_type="transformer", dtype="float32", file_size=1024, layer_type="huggingface", component_name="bert-base-uncased")

#         # Create another Compatability object
#         component2 = Compatability(model_type="resnet50", dtype="int8", file_size=512, layer_type="torchvision", component_name="resnet50-pretrained")

#         # Create an Architecture object and add components to it
#         architecture1 = Architecture(architecture="nlp")
#         architecture1.add_implementation("transformer", component1)

#         architecture2 = Architecture(architecture="vision")
#         architecture2.add_implementation("resnet50", component2)

#         # Create a Domain object and add architectures to it
#         domain = Domain(domain_name="models")
#         domain.add_architecture("nlp", architecture1)
#         domain.add_architecture("vision", architecture2)

#         # Expected dictionary structure
#         expected_dict = {
#             "models.nlp.transformer": {"model_type": "transformer", "dtype": "float32", "file_size": 1024, "layer_type": "huggingface", "component_name": "bert-base-uncased"},
#             "models.vision.resnet50": {"model_type": "resnet50", "dtype": "int8", "file_size": 512, "layer_type": "torchvision", "component_name": "resnet50-pretrained"},
#         }
#         self.allowed_keys = {"dtype", "file_size", "disk_path", "layer_type", "component_name", "custom_slot_1", "custom_slot_2"}

#         # Convert the domain to a dictionary and assert equality
#         actual_dict = domain.to_dict()
#         self.assertEqual(actual_dict, expected_dict)
