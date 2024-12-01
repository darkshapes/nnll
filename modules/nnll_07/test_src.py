
from .src import Domain, Architecture, Component


def main():
    # Create a Domain instance
    domain = Domain("ml")

    # Create an Architecture instance
    architecture1 = Architecture("gpt-3")
    architecture2 = Architecture("bert")

    # Create Component instances for GPT-3
    component1_gpt3 = Component("text_generation", dtype="float32", file_size=1024, library="transformers")
    component2_gpt3 = Component("language_modeling", dtype="float16", file_size=512)

    # Create Component instances for BERT
    component1_bert = Component("text_classification", dtype="float32", file_size=768)
    component2_bert = Component("named_entity_recognition", dtype="float16")

    # Add components to architectures
    architecture1.add_component("text_generation", component1_gpt3)
    architecture1.add_component("language_modeling", component2_gpt3)

    architecture2.add_component("text_classification", component1_bert)
    architecture2.add_component("named_entity_recognition", component2_bert)

    # Add architectures to domain
    domain.add_architecture("gpt-3", architecture1)
    domain.add_architecture("bert", architecture2)

    # Convert the entire structure to a dictionary and print it
    result_dict = domain.to_dict()
    print(result_dict)


if __name__ == "__main__":
    main()


# # # Example usage
# # domain_ml = Domain("ml")
# # arch_sdxl_base = Architecture("sdxl-base")
# # comp_unet = Component("unet", dtype="float32", file_size=1024, component_name="base")

# # arch_sdxl_base.add_component(comp_unet.model_type, comp_unet)
# # domain_ml.add_architecture(arch_sdxl_base.architecture, arch_sdxl_base)

# # model_index_dict = domain_ml.to_dict()
# # print(model_index_dict)

# # Create a domain
# domain_ml = Domain("ml")

# # Create architectures within the domain
# arch_sdxl_base = Architecture("sdxl-base")
# arch_auraflow = Architecture("auraflow")
# arch_flux = Architecture("flux")

# # Create components within architectures
# comp_unet = Component("unet", dtype="float32", file_size=1024, library="diffusers")
# comp_vae = Component("vae", dtype="float32", file_size=512, library="diffusers")
# comp_lora = Component("lora", dtype="float32", file_size=256, library="diffusers")

# # Add components to architectures
# arch_sdxl_base.add_component(comp_unet.component_name, comp_unet)
# arch_auraflow.add_component(comp_vae.component_name, comp_vae)
# arch_flux.add_component(comp_lora.component_name, comp_lora)

# # Add architectures to domain
# domain_ml.add_architecture(arch_sdxl_base.architecture, arch_sdxl_base)
# domain_ml.add_architecture(arch_auraflow.architecture, arch_auraflow)
# domain_ml.add_architecture(arch_flux.architecture, arch_flux)

# # Serialize the domain to a dictionary for storage or transmission
# model_index_dict = domain_ml.to_dict()
# print(model_index_dict)

# Test the Domain, Architecture, and Component classes
