
class Domain:
    """
    Domains:
    ml : Publicly released machine learning models with an identifier in the database
    info : Metadata with an identifier in the database
    dev : Any pre-release or under evaluation items without an identifier in an expected format
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
    Known generative and deep learning architectures. See tuning.json for more information
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
    Modalities, contents, techniques, or purposes of an identified model that the system should know of.
    """

    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.allowed_keys = {"dtype", "file_size", "library", "component_name"}
        for key, value in kwargs.items():
            if key in self.allowed_keys:
                setattr(self, key, value)

    def to_dict(self):
        result = {"model_type": self.model_type}

        for key in self.allowed_keys:
            if hasattr(self, key):
                result.setdefault(key, getattr(self, key))

        return result
