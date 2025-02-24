# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

"""
Identification system for neural network models
`domain_name`  [ml/info/dev] see `domain class` for details
`architecture` the family and version (stable diffusion 3.5, lumina next)
`components`   attributes and process stage (lora, unet)

# Create a domain
`domain_ml = Domain("ml")`

# Create architectures within the domain
```
arch_sdxl_base = Architecture("sdxl-base")
arch_auraflow = Architecture("auraflow")
arch_flux = Architecture("flux")
```

# Create components within architectures
```
comp_unet = Component("unet", dtype="float32", file_size=1024, layer_type="diffusers")
comp_vae = Component("vae", dtype="float32", file_size=512, layer_type="diffusers")
comp_lora = Component("lora", dtype="float32", file_size=256, layer_type="diffusers")
```

# Add components to architectures
```
arch_sdxl_base.add_component(comp_unet.component_name, comp_unet)
arch_auraflow.add_component(comp_vae.component_name, comp_vae)
arch_flux.add_component(comp_lora.component_name, comp_lora)
```

# Add architectures to domain
domain_ml.add_architecture(arch_sdxl_base.architecture, arch_sdxl_base)
domain_ml.add_architecture(arch_auraflow.architecture, arch_auraflow)
domain_ml.add_architecture(arch_flux.architecture, arch_flux)

# Serialize the domain to a dictionary for storage or transmission
model_index_dict = domain_ml.to_dict()
print(model_index_dict)

"""


class Domain:
    """
    ### Domains:
    Valid domains can be anything, though our guidelines follow\n
    ***ml*** : Publicly released machine learning models with an identifier in the database\n
    ***info*** : Metadata with an identifier in the database\n
    ***dev*** : Any pre-release or under evaluation items without an identifier in an expected format\n

    :method add_architecture: create a sub-class of the domain
    :method to_dict(): flatten the class structure
    """

    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.architectures = {}

    def add_architecture(self, architecture_name, architecture_obj):
        self.architectures[architecture_name] = architecture_obj

    def to_dict(self):
        flat_dict = {}
        for arc_name, arc_obj in self.architectures.items():
            path = f"{self.domain_name}.{arc_name}"
            flat_dict.update(arc_obj.to_dict(path))
        return flat_dict


class Architecture:
    """
    Known generative and deep learning architectures.\n
    model_forms.json contains the lengthy key list of supported architectures\n

    :method add_component: create a sub-class of the architecture
    :method to_dict(): flatten the class structure
    """

    def __init__(self, architecture):
        self.architecture = architecture
        self.components = {}

    def add_component(self, model_type, component_obj):
        self.components[model_type] = component_obj

    def to_dict(self, prefix):
        flat_dict = {}
        for comp_name, comp_obj in self.components.items():
            path = f"{prefix}.{comp_name}"
            flat_dict[path] = comp_obj.to_dict()
        return flat_dict


class Component:
    """
    Specifics of modalities, contents, techniques, or purposes of an identified model that effect processing.\n
    This enables us to filter, organize, and prepare files, allowing automated workflow construction\n
    :param model_type: Classification of the file, what purpose this model serves as a whole,  (eg: unet, vae, lora)\n
    :param kwargs: Named values from one of the following
    ***file_size*** : The total size in **bytes** of the file\n
    ***disk_path*** : The full location of the file\n
    ***file_name*** : The basename of the file\n
    ***extension*** : The last file extension in the filename\n
    ***dtype*** : The model datatype format (if applicable, to know if and how precision can be lowered)\n
    ***component_type*** : Sub-components of the model_type as a list
    ***component_name*** : A specific title or technique of a component/model_type
    ***layer_type*** : The format and compatibility of the model structure, including but not limited to\n
    - `pytorch`
    - `diffusers`
    - `compvis`
    (Note: Future functionality should include dynamically adding permanent custom attributes)
    :method to_dict(): flatten the class structure
    """

    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        self.allowed_keys = {
            "dtype",
            "file_size",
            "layer_type",
            "component_type",
            "component_name",
            "file_extension",
            "file_name",
            "disk_path",
        }
        for key, value in kwargs.items():
            if key not in self.allowed_keys:
                raise KeyError(f"Valid attributes can only be one of the following : {(k for k in self.allowed_keys)}")
            else:
                setattr(self, key, value)

    def to_dict(self):
        result = {"model_type": self.model_type}

        for key in self.allowed_keys:
            if hasattr(self, key):
                result.setdefault(key, getattr(self, key))

        return result
