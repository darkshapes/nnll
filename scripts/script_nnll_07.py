

from modules.nnll_07.src import Architecture, Domain, Component

# domain_ml = Domain("ml")
# arch_sdxl_base = Architecture("sdxl-base")
# comp_unet = Component("unet", dtype="float32", file_size=1024, component_name="base")

# arch_sdxl_base.add_component(comp_unet.model_type, comp_unet)
# domain_ml.add_architecture(arch_sdxl_base.architecture, arch_sdxl_base)

# model_index_dict = domain_ml.to_dict()
# print(model_index_dict)

# Create a domain
domain_ml = Domain("ml")

# Create architectures within the domain
arch_sdxl_base = Architecture("sdxl-base")
arch_auraflow = Architecture("auraflow")
arch_flux = Architecture("flux")

# Create components within architectures
comp_unet = Component("unet", dtype="float32", file_size=1024, library="diffusers")
comp_vae = Component("vae", dtype="float32", file_size=512, library="diffusers")
comp_lora = Component("lora", dtype="float32", file_size=256, library="diffusers")

# Add components to architectures
arch_sdxl_base.add_component(comp_unet.component_name, comp_unet)
arch_auraflow.add_component(comp_vae.component_name, comp_vae)
arch_flux.add_component(comp_lora.component_name, comp_lora)

# Add architectures to domain
domain_ml.add_architecture(arch_sdxl_base.architecture, arch_sdxl_base)
domain_ml.add_architecture(arch_auraflow.architecture, arch_auraflow)
domain_ml.add_architecture(arch_flux.architecture, arch_flux)

# Serialize the domain to a dictionary for storage or transmission
model_index_dict = domain_ml.to_dict()
print(model_index_dict)
