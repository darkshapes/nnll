### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Any, Dict, Union, List, Generic, TypeVar
import urllib.parse
from nnll_01 import debug_monitor

T = TypeVar("T")

from pydantic import create_model
from typing import Dict, Any, Union
from dataclasses import asdict


@dataclass
class Info:
    """
    Static global neural network attributes, metadata with an identifier in the database\n
    :param gen_kwargs: OEM arguments to pass to the generator
    :param init_kwargs: OEM arguments to pass to constructor
    :param layer_256: Canonical hash calculation for list of model layer names, if applicable
    :param repo: A dedicated remote origin
    :param tasks: Supported modalities
    :param weight_map: Remote location of the weight map for the model
    """

    dep_pkg: Optional[List[str]] = None
    module_path: Optional[List[str]] = None
    gen_kwargs: Optional[Dict[str, Union[float, int, Dict[str, int | str], Callable]]] = None
    init_kwargs: Optional[Dict[str, Union[float, int, Dict[str, int | str], Callable]]] = None
    layer_256: Optional[List[str]] = None
    repo: Optional[str] = None
    solver_kwargs: Optional[Dict[str, Union[float, int, Dict[str, int | str], Callable]]] = None
    tasks: Optional[List[str]] = None
    weight_map: Optional[urllib.parse.ParseResult] = None


@dataclass
class Ops:
    """
    Varying global neural network attributes, algorithms, optimizations and procedures on models\n
    info: str  # Immutable metadata with an identifier in the database\n
    dev: str  # Any pre-release or under evaluation items without an identifier in an expected format\n
    :param info:  Static global neural network attributes, metadata with an identifier in the database\n
    `info` domain attributes
    :param init_kwargs: OEM arguments to pass to the constructor
    :param gen_kwargs: OEM arguments to pass to the generator
    :param layer_256: Canonical hash calculation for list of model layer names, if applicable
    :param name: A specific title or technique identifier
    :param solver_kwargs: OEM arguments to pass to constructor function
    :param repo: A dedicated remote origin
    """

    init_kwargs: Optional[Dict[str, int | str | float | list]] = None
    gen_kwargs: Optional[Dict[str, int | str | float | list]] = None
    dep_pkg: Optional[List[str]] = None
    module_path: Optional[List[str]] = None
    repo: Optional[str] = None


@dataclass
class Model:
    """
    Static local neural network layers. Publicly released machine learning models with an identifier in the database\n
    :param dtype: Model precision (ie F16,F32,F8_E4M3,I64)
    :param file_256: Canonical hash calculation for known model files
    :param file_ext: The last file extension in the filename
    :param file_name: The basename of the file
    :param file_path: Absoulte location of the file on disk
    :param file_size: Total size of the file in bytes
    :param layer_type: The format and compatibility of the model structure (e.g., 'diffusers')
    """

    dtype: Optional[str] = None
    file_256: Optional[str] = None
    file_ext: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[Path] = None
    file_size: Optional[int] = None
    layer_type: Optional[str] = None


@dataclass
class Dev:
    """
    Varying local neural network layers, in-training, pre-release, items under evaluation, likely in unexpected formats\n
    """

    dependency_pkg: Optional[list[str]] = None
    dtype: Optional[str] = None
    file_256: Optional[str] = None
    file_ext: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[Path] = None
    file_size: Optional[int] = None
    gen_kwargs: Optional[Dict[str, Union[float, int, Callable]]] = None
    layer_256: Optional[str] = None
    layer_type: Optional[str] = None
    lora_kwargs: Optional[str] = None
    module_path: Optional[list[str]] = None
    pipe_kwargs: Optional[Dict[str, Union[float, int, Callable]]] = None
    repo: Optional[str] = None
    solver_kwargs: Optional[Dict[str, Union[float, int, Callable]]] = None
    tasks: Optional[List[str]] = None
    weight_map: Optional[urllib.parse.ParseResult] = None

    # layer_256: Optional[str] = None
    # tasks: Optional[List[str]] = None
    # weight_map: Optional[urllib.parse.ParseResult] = None
    # gen_kwargs: Optional[dict[str, Union[float, int, Callable]]] = None
    # solver_kwargs: Optional[dict[str, Union[float, int, Callable]]] = None
    # :param stage: Where item fits in a chain


def build_model(fields: Dict[str, Any]):
    """Dynamically create `Compatability` class attributes\n
    :param fields: Title of the field in `Compatibility`
    """
    field = {fields: (Union[Info, Model, Ops, Dev], ...)}
    return create_model("Compatibility", **field)


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
        for comp_name, comp_obj in self.compatibility.items():
            path = f"{prefix}"  # .{comp_name}"
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
    **to_dict** Flatten the Architeture class structure
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

    @debug_monitor
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


def add_mir_fields(domain: str, **kwargs):
    """_summary_\n
    :param domain: Class type field constructor
    :raises ValueError: An invalid domain type was entered
    :return: A class of the designated type constructed with all fields added
    """
    if domain.lower() == "info":
        return Info(**{k: v for k, v in kwargs.items() if k in Info.__dataclass_fields__})  # pylint: disable=no-member
    elif domain.lower() == "model":
        return Model(**{k: v for k, v in kwargs.items() if k in Model.__dataclass_fields__})  # pylint: disable=no-member
    elif domain.lower() == "ops":
        return Ops(**{k: v for k, v in kwargs.items() if k in Ops.__dataclass_fields__})  # pylint: disable=no-member
    elif domain.lower() == "dev":
        return Dev(**{k: v for k, v in kwargs.items() if k in Dev.__dataclass_fields__})  # pylint: disable=no-member
    else:
        raise ValueError(f"Unsupported domain: {domain}")


# from pydantic import create_model
# def build_model(fields: Dict[str, Any]):
#     return create_model("Compatibility", **fields)
#   data = {compatibility: add_mir_fields(domain=domain.lower(), **kwargs)}
#     from nnll_01 import nfo

#     Compatibility = build_model({f"{compatibility}": (Union[Info, Model, Ops, Dev], ...)})
#     compat = Compatibility(**data)


def add_mir_entry(domain: str, arch: str, series: str, compatibility: str, **kwargs) -> None:
    """Define a new Machine Intelligence Resource\n
    :param domain: Broad name of the type of data (model/ops/info/dev)
    :param arch: Common name of the neural network structure being referenced
    :param series: Specific release name or technique
    :param compatibility: Details about purpose, tasks
    :param kwargs: Specific key/value data related to location and execution
    """

    domain_info = Domain(domain.lower())
    arch_name = Architecture(arch.lower())
    impl_sdxl = Series(series.lower())
    # kwargs.setdefault("name", compatibility.lower())
    data = {compatibility: add_mir_fields(domain=domain.lower(), **kwargs)}

    Compatibility = build_model(compatibility)

    class CompatibilityModel(Compatibility):
        def to_dict(self, _) -> Dict[str, Any]:
            return {key: value for key, value in self.__dict__.items() if value is not None}

    # from nnll_01 import nfo

    compat = CompatibilityModel(**data)
    # nfo(compat)
    # compat = Compatibility(compatibility.lower(), add_mir_fields(domain=domain.lower(), **kwargs))
    impl_sdxl.add_compat(compatibility, compat)
    arch_name.add_impl(impl_sdxl.series, impl_sdxl)
    domain_info.add_arch(arch_name.architecture, arch_name)
    return domain_info.to_dict()


if __name__ == "__main__":
    import argparse
    from nnll_60 import JSONCache, CONFIG_PATH_NAMED

    config_file = JSONCache(CONFIG_PATH_NAMED)
    parser = argparse.ArgumentParser(description="MIR database manager")
    parser.add_argument("-r", "--remove", action="store_true", help="Remove an item from the database (currently not implemented)")
    parser.add_argument("-d", "--domain", type=str, help=" Broad name of the type of data (model/ops/info/dev)")
    parser.add_argument("-a", "--arch", type=str, help=" Common name of the neural network structure being referenced")
    parser.add_argument("-s", "--series", type=str, help="Specific release title or technique")
    parser.add_argument("-c", "--compatibility", type=str, help="Details about purpose, tasks")
    parser.add_argument("-k", "--kwargs", type=str, help="Keyword arguments to pass to function constructors by default")

    args = parser.parse_args()

    @config_file.decorator
    def read_data(data: Dict[str, int | float | str | list] = None) -> dict:
        """Update MIR file with new entry
        :param data: existing dictionary
        """
        data.update_cache(
            add_mir_entry(domain=args.domain, arch=args.arch, series=args.series, compatibility=args.compatibility, **args.kwargs),
        )

    read_data()
