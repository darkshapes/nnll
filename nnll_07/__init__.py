### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from dataclasses import dataclass
from typing import Optional, Any, Dict

from nnll_01 import debug_monitor


class Domain:
    """
    ### Domains:
    Valid domains can be anything, though our guidelines follow\n
    ***ml*** : Publicly released machine learning models with an identifier in the database\n
    ***info*** : Metadata with an identifier in the database\n
    ***dev*** : Any pre-release or under evaluation items without an identifier in an expected format\n
    """

    @debug_monitor
    def __init__(self, domain_name: str) -> None:
        self.domain_name = domain_name
        self.architectures = {}

    @debug_monitor
    def add_architecture(self, architecture_name: str, architecture_obj: Any) -> None:
        """Create a sub-class of Domain"""
        self.architectures[architecture_name] = architecture_obj

    @debug_monitor
    def to_dict(self) -> Dict[str, Any]:
        """Flatten the Domain class structure"""
        flat_dict = {}
        for arc_name, arc_obj in self.architectures.items():
            path = f"{self.domain_name}.{arc_name}"
            flat_dict.update(arc_obj.to_dict(path))
        return flat_dict


class Architecture:
    """
    Known generative and deep learning architectures.\n
    model_forms.json contains the lengthy key list of supported architectures\n
    """

    @debug_monitor
    def __init__(self, architecture: str) -> None:
        self.architecture = architecture
        self.components = {}

    def add_component(self, model_type: str, component_obj: str) -> None:
        """Add_component: create a sub-class of the architecture"""
        self.components[model_type] = component_obj

    def to_dict(self, prefix: str) -> Dict[str, Any]:
        """:Flatten the Architecture class structure"""
        flat_dict = {}
        for comp_name, comp_obj in self.components.items():
            path = f"{prefix}.{comp_name}"
            flat_dict[path] = comp_obj.to_dict()
        return flat_dict


@dataclass
class Component:
    """
    Specifics of modalities, contents, techniques, or purposes of an identified model.

    This class encapsulates the attributes required to define a component used in a machine
    learning model. Only attributes with non-None values are included during serialization.

    :param model_type: Classification of the file or component (e.g., 'unet', 'vae', 'lora').
    :type model_type: str
    :param dtype: The model datatype format, indicating precision (optional).
    :type dtype: Optional[str]
    :param file_size: The total size in bytes of the file (optional).
    :type file_size: Optional[int]
    :param layer_type: The format and compatibility of the model structure (e.g., 'diffusers') (optional).
    :type layer_type: Optional[str]
    :param component_type: Sub-components of the model type as a list or any additional type information (optional).
    :type component_type: Optional[Any]
    :param component_name: A specific title or technique identifier for the component (optional).
    :type component_name: Optional[str]
    :param file_extension: The last file extension in the filename (optional).
    :type file_extension: Optional[str]
    :param file_name: The basename of the file (optional).
    :type file_name: Optional[str]
    :param disk_path: The full location of the file on disk (optional).
    :type disk_path: Optional[str]
    """

    model_type: str
    component_name: Optional[str] = None
    component_type: Optional[Any] = None
    disk_path: Optional[str] = None
    dtype: Optional[str] = None
    file_extension: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    layer_type: Optional[str] = None

    @debug_monitor
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Component to a dictionary, including only attributes that are not None.

        :return: A dictionary representation of the Component.
        :rtype: Dict[str, Any]
        """
        return {key: value for key, value in vars(self).items() if value is not None}
