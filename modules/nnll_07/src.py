
"""
Identification system for neural network models
`domain_name`  [ml/info/dev] see `domain class` for details
`architecture` the family and version (stable diffusion 3.5, lumina next)
`components`   attributes and process stage (lora, unet)
"""


class Domain:
    """
    ### Domains:
    Valid domains can be anything, though our guidelines are \n
    ***ml*** : Publicly released machine learning models with an identifier in the database\n
    ***info*** : Metadata with an identifier in the database\n
    ***dev*** : Any pre-release or under evaluation items without an identifier in an expected format\n
    methods\n
    - `add_architecture()`,
    - `to_dict()`
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
    model_forms.json contains the lengthy key list of supported architectures
    #### ***Component*** The methodology classifying the file (eg: unet, vae, lora)
    methods\n
    - `add_component()`,
    - `to_dict()`
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
    This enables us to filter, organize, and prepare files, allowing automated workflow construction
    ***model_type*** : What purpose this model serves as a whole\n
    ***disk_size*** : The total size in **bytes** of the file\n
    ***disk_path*** : The full location of the file\n
    ***file_name*** : The basename of the file\n
    ***extension*** : The last file extension in the filename\n
    ***dtype*** : The model datatype format (if applicable, to know if and how precision can be lowered)\n
    ***component_type*** : Sub-components of the model_type as a list
    ***component_name*** : A specific title or technique of a component/model_type
    ***layer_type*** : The format and compatability of the model structure, including but not limited to\n
    - `pytorch`
    - `diffusers`
    - `compvis`
    (Note: Future functionality should include dynamically adding permament custom attributes)

    """

    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        self.allowed_keys = {"dtype", "disk_size", "layer_type", "component_type", "component_name", "file_extension", "file_name", "disk_path", }
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
