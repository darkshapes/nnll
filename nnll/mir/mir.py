# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from logging import INFO, Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, create_model

nfo_obj = Logger(INFO)
nfo = nfo_obj.info

T = TypeVar("T")

PackageIndex = int
TimestepType = List[int]  # never none
ParameterField = Union[str, bool, int, float, TimestepType]  # never none

PkgType = Dict[
    PackageIndex,  # Outer key is an integer (e.g., 0)
    Dict[ParameterField, Union[Dict, ParameterField]],
]

HashType = Optional[Union[List[str], Optional[Dict[str, Union[Any]]]]]


class Info(BaseModel):
    """
    Static global neural network attributes, metadata with an identifier in the database\n
    :param file_256: Canonical hash calculation for known model files
    :param gen_kwargs: OEM arguments to pass to the generator
    :param init_kwargs: OEM arguments to pass to constructor
    :param module_alt: Third-party library module support for the resource
    :param layer_256: Canonical hash calculation for list of model layer names, if applicable
    :param repo: A dedicated remote origin
    :param repo_alt: Secondary remote sources
    :param scheduler_kwargs: OEM arguments to pass to constructor function
    :param scheduler: OEM noise scheduler (mir:type str)
    :param tasks: Supported modalities
    :param weight_map: Remote location of the weight map for the model
    """

    repo: Optional[str] = None
    pkg: Optional[PkgType] = None
    file_256: Optional[HashType] = None
    layer_256: Optional[HashType] = None
    file_b3: Optional[HashType] = None
    layer_b3: Optional[HashType] = None
    identifier: Optional[Union[List[int], int, str]] = None  # numbers preceding str


class Ops(BaseModel):
    """
    Varying global neural network attributes, algorithms, optimizations and procedures on models\n
    info: str  # Immutable metadata with an identifier in the database\n
    dev: str  # Any pre-release or under evaluation items without an identifier in an expected format\n
    :param info:  Static global neural network attributes, metadata with an identifier in the database\n
    `info` domain attributes
    :param dtype: Model datatype (ie F16,F32,F8_E4M3,I64) name if applicable
    :param gen_kwargs: OEM arguments to pass to the generator
    :param init_kwargs: OEM arguments to pass to the constructor
    :param repo: A dedicated remote origin
    :param scheduler_kwargs: OEM arguments to pass to constructor function
    :param dtype: Alternate datatype name
    """

    pkg: PkgType
    repo: Optional[List[str]] = None


class Model(BaseModel):
    """
    Static local neural network layers. Publicly released machine learning models with an identifier in the database\n
    :param dtype: Model datatype (ie F16,F32,F8_E4M3,I64) name if applicable
    :param file_ext: The last file extension in the filename
    :param file_name: The basename of the file
    :param file_path: Absolute location of the file on disk
    :param file_size: Total size of the file in bytes
    :param layer_type: The format and compatibility of the model structure (e.g., 'diffusers')
    """

    dtype: Optional[str] = None
    file_ext: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[Path] = None
    file_size: Optional[int] = None
    format: Optional[str] = None
    layer_type: Optional[str] = None


@dataclass
class Dev(Info, Ops, Model):
    """
    Varying local neural network layers, in-training, pre-release, items under evaluation, likely in unexpected formats\n
    Inheriting attributes from Info, Ops, and Model to reduce duplication.
    """

    # Experimental/Optional/Deprecated, maybe useful later
    stage: str  # Where item fits in a chain
    dtype: Optional[str] = None
    dep_pkg: Optional[Dict[str, list[str]]] = None
    gen_kwargs: Optional[Dict[str, Any]] = None
    lora_kwargs: Optional[str] = None
    module_alt: Optional[List[str]] = None
    module_path: Optional[list[str]] = None
    repo_pkg: Optional[List[str]] = None
    requires: Optional[Dict[str, list[str]]] = None
    scheduler_alt: Optional[str] = None
    scheduler_kwargs_alt: Optional[Dict[str, Any]] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    scheduler: Optional[str] = None
    tasks: Optional[List[str]] = None
    weight_map: Optional[Union[urllib.parse.ParseResult, str]] = None
    # gen_kwargs: Optional[Dict[str, int | str | float | list]] = None
    # init_kwargs: Optional[Dict[str, int | str | float | list]] = None


def add_mir_fields(domain: str, **kwargs):
    """Build MIR dataclasses for each domain type\n
    :param domain: Class type field constructor
    :raises ValueError: An invalid domain type was entered
    :return: A class of the designated type constructed with all fields added
    """
    if domain.lower() == "info":
        return Info(**{k: v for k, v in kwargs.items() if k in Info.__pydantic_fields__ and v is not None})  # pylint: disable=no-member
    elif domain.lower() == "model":
        return Model(**{k: v for k, v in kwargs.items() if k in Model.__pydantic_fields__ and v is not None})  # pylint: disable=no-member
    elif domain.lower() == "ops":
        return Ops(**{k: v for k, v in kwargs.items() if k in Ops.__pydantic_fields__ and v is not None})  # pylint: disable=no-member
    elif domain.lower() == "dev":
        return Dev(**{k: v for k, v in kwargs.items() if k in Dev.__dataclass_fields__ and v is not None})  # pylint: disable=no-member
    raise ValueError(f"Unsupported domain: {domain}")


def build_comp(comp: str, domain: str, kwargs: dict) -> Callable:
    """Create a dynamic Compatibility class and return the defined class object with a flattening function\n
    :param comp: Name for the class field (and subsequent flattened `key`)
    :param domain: Class type for the `value` data
    :param kwargs: Key/value data pairs within `value`
    :return: A Compatibility object with `comp` name mapped to `kwargs` data
    """
    field = {
        comp: (Union[Info, Model, Ops, Dev], ...),
    }
    data = {}
    data.setdefault(comp, add_mir_fields(domain=domain, **kwargs))

    DynamicModel = create_model("Compatibility", **field)  # pylint: disable=invalid-name

    class Compatibility(DynamicModel):  # pylint: disable=too-few-public-methods
        """Dynamically created model attributes (to create key structure)"""

        def to_dict(self, _) -> Dict[str, Any]:
            """Flatten the Compatibility class structure\n
            :return: A dictionary of the structure
            """
            setattr(self, comp, getattr(self, comp).__dict__)  # add the data to an attribute if its not already made
            return {
                comp_name: {
                    inner_key: inner_value
                    for inner_key, inner_value in comp_value.items()
                    if inner_value is not None  # Comment to force formatting
                }
                for comp_name, comp_value in self.__dict__.items()
                if isinstance(comp_value, dict)
            }

    return Compatibility(**data)


class Series:
    """
    Specific seriess of generative and deep learning architectures.\n
    The release name of models, in short.
    mir.json contains the lengthy key list of supported seriess\n
    **add_compat** Add a Compatibility object to this class
    **to_dict** flatten the compatibility object
    """

    # @debug_monitor
    def __init__(self, series: str) -> None:
        """Constructor"""
        self.series = series
        self.compatibility = defaultdict(dict)
        self.flat_dict = defaultdict(dict)

    def add_compat(self, compat_label: str, compat_obj: Dict[str, int | float | list | str]) -> None:
        """Add compatibility: Attribute an object to a sub-class of the Series"""
        self.compatibility[compat_label] = compat_obj

    def to_dict(self, prefix: str) -> Dict[str, Any]:  # , prefix: str = "compatibility")
        """Flatten the Architecture class structure\n
        :param prefix: Prepended identifying tag"""
        for _comp_name, comp_obj in self.compatibility.items():
            path = f"{prefix}"  # .{comp_name}" # so that we can combine last keys into one dict
            self.flat_dict[path].update(comp_obj.to_dict(path))  # path
        return self.flat_dict


class Architecture:
    """
    Known generative and deep learning architecture.\n
    :param art: Autoregressive transformer, typically LLMs
    :param dit: Diffusion transformer, typically Vision Synthesis
    :param lora: Low-Rank Adapter (may work with dit or transformer)
    :param unet: Unet diffusion structure
    :param vae: Variational Autoencoder, roughly

    **add_impl** Add an Series object to the Architecture
    **to_dict** Flatten the Architecture class structure
    """

    # @debug_monitor
    def __init__(self, architecture: str) -> None:
        """Constructor"""
        self.architecture = architecture
        self.series = defaultdict(dict)
        self.flat_dict = defaultdict(dict)

    def add_impl(self, impl_label: str, impl_obj: Series) -> None:
        """Add_component: Attribute an object to a sub-class of the Architecture"""
        self.series[impl_label] = impl_obj

    def to_dict(self, prefix: str) -> Dict[str, Any]:
        """Flatten the Architecture class structure\n
        :param prefix: Prepended identifying tag"""
        for comp_name, comp_obj in self.series.items():
            path = f"{prefix}.{comp_name}"
            self.flat_dict.update(comp_obj.to_dict(path))
        return self.flat_dict


class Domain:
    """
    Define a set of AI/ML related data\n
    :param dev: `Dev()`
    :param model: `Model()`
    :param Ops: `Ops()`
    :param info: `Info()`

    **add_arch** Create a sub-class of Domain\n
    **to_dict** Flatten the Domain class structure
    """

    dev: Dev  # Pre-release or under evaluation items without an identifier in an expected format
    info: Info  # Metadata of layer names or settings with an identifier in the database
    model: Model  # Model weight specifics of shifting locations and practical dimensions
    ops: Ops  # References to specific optimization or manipulation techniques

    # @debug_monitor
    def __init__(self, domain_name: str) -> None:
        """Constructor"""
        self.domain_name = domain_name
        self.architectures = defaultdict(dict)
        self.flat_dict = defaultdict(dict)

    # @debug_monitor
    def add_arch(self, arch_label: str, arch_obj: Architecture) -> None:
        """
        Add an architecture to subclass this Domain\n
        :param architecture_name: A valid architecture type
        :param architecture_obj: Data to store
        """
        self.architectures[arch_label] = arch_obj

    # @debug_monitor
    def to_dict(self) -> Dict[str, Any]:
        """Flatten the Architecture class structure\n
        :return: A dictionary of the structure
        """
        for arc_name, arc_obj in self.architectures.items():
            path = f"{self.domain_name}.{arc_name}"
            self.flat_dict.update(arc_obj.to_dict(path))
        return self.flat_dict


def mir_entry(domain: str, arch: str, series: str, comp: str, **kwargs) -> None:
    """Define a new Machine Intelligence Resource\n
    :param domain: Broad name of the type of data (model/ops/info/dev)
    :param arch: Common name of the neural network structure being referenced
    :param series: Specific release name or technique
    :param comp: Details about purpose, tasks
    :param kwargs: Specific key/value data related to location and execution
    """

    domain_inst = Domain(domain.lower())
    arch_inst = Architecture(arch.lower())
    series_inst = Series(series.lower())
    comp_inst = build_comp(comp, domain, kwargs)
    series_inst.add_compat(comp, comp_inst)
    arch_inst.add_impl(series_inst.series, series_inst)
    domain_inst.add_arch(arch_inst.architecture, arch_inst)
    return domain_inst.to_dict()


# def create_model_tag(model_header,metadata_dict):
#         parse_file = parse_model_header(model_header)
#         reconstructed_file_path = os.path.join(disk_path,each_file)
#         attribute_dict = metadata_dict | {"disk_path": reconstructed_file_path}
#         file_metadata = parse_file | attribute_dict
#         index_tag = create_model_tag(file_metadata)
#


def main():
    """Add a single entry to MIR database\n"""
    import argparse

    from nnll.mir.json_cache import MIR_PATH_NAMED, JSONCache  # pylint:disable=no-name-in-module

    parser = argparse.ArgumentParser(description="MIR database manager")
    parser.add_argument("-r", "--remove", action="store_true", help="Remove an item from the database (currently not implemented)")
    parser.add_argument("-d", "--domain", type=str, help=" Broad name of the type of data (model/ops/info/dev)")
    parser.add_argument("-a", "--arch", type=str, help=" Common name of the neural network structure being referenced")
    parser.add_argument("-s", "--series", type=str, help="Specific release title or technique")
    parser.add_argument("-c", "--compatibility", type=str, help="Details about purpose, tasks")
    parser.add_argument("-k", "--kwargs", type=str, help="Keyword arguments to pass to function constructors by default")

    args = parser.parse_args()

    def add_data() -> dict:
        """Update MIR file with new entry
        :param data: existing dictionary
        """
        mir_file = JSONCache(MIR_PATH_NAMED)

        @mir_file.decorator
        def _read_data(data: Dict[str, int | float | str | list] = None):
            return data

        mir_db = _read_data()
        mir_db.update_cache(
            mir_entry(domain=args.domain, arch=args.arch, series=args.series, comp=args.compatibility, **args.kwargs),
        )

    add_data()


if __name__ == "__main__":
    main()
