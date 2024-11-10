import os
import json
import tomllib
from glob import glob
from functools import cache, cached_property, partial, total_ordering
from rich.logging import RichHandler
from rich.console import Console
from rich.logging import RichHandler

from pydantic import BaseModel, Field, ConfigDict

config_source_location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Logger:
    log_level = "INFO"
    msg_init = None
    handler = RichHandler(console=Console(stderr=True))

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)  # same as print
        handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda s: s.replace('_', '-')
    )

class LocationConfig(ConfigModel):
    clients: str = "clients"
    nodes: str = "nodes"
    flows: str = "flows"
    input: str = "input"
    output: str = "output"
    models: str = "models"

class Contents:
    location: LocationConfig = Field(default_factory=LocationConfig)

    """
    ## CLASS Contents
    ##### IMPORT from get import Contents
    ##### METHODS get_default, get_path_contents, get_path, model_indexer, t2i_pipe, node_tuner, sys_cap
    ##### VARIABLES config_source_location
    ##### PURPOSE find source directories and data
    ##### OUTPUT a dict of keys, a dict of files, a path to a file
    ##### RETURN FORMAT: {key:},
    ##### SYNTAX
    ```
            config.get_default(filename with no extension, key)               (!cannot find sub-keys on its own)
            config.get_path_contents("string_to_folder.string_to_sub_folder") (see config/config.json, config/directories.json)
            config.get_path("filename") or config.get_path("string_to_folder.filename")

            os.path.join(config_source_location,filename)
    ```
    from sdbx.config import model_indexer
    from sdbx.indexer import ModelType

    """

    @cached_property
    def _path_dict(self):
        root = {
            n: os.path.join(self.path, p) for n, p in dict(self.location).items() # see self.location for details
        }

        for n, p in dict(self.location).items():
            if ".." in p:
                raise Exception("Cannot set location outside of config path.")

        models = {f"models.{name}": os.path.join(root["models"], name) for name in self.get_default("directories", "models")}

        return {**root, **models}

    def load_data(self, path):
        _, ext = os.path.splitext(path)
        loader, mode = (tomllib.load, "rb") if ext == ".toml" else (json.load, "r")
        with open(path, mode) as f:
            try:
                fd = loader(f)
            except (tomllib.TOMLDecodeError, json.decoder.JSONDecodeError) as e:
                raise SyntaxError(f"Couldn't read file {path}") from e
        return fd

    @cached_property
    def _defaults_dict(self):
        d = {}
        glob_source = partial(glob, root_dir=config_source_location)
        for filename in glob_source("*.toml") + glob_source("*.json"):
            fp = os.path.join(config_source_location, filename)
            name, _ = os.path.splitext(filename)
            d[name] = self.load_data(fp)

        return d

    def get_default(self, name, prop):
        return self._defaults_dict[name][prop]

    @cached_property
    def model_indexer(self):
        from indexer import ModelIndexer
        return ModelIndexer()

    @cached_property
    def t2i_pipe(self):
        from compute import T2IPipe
        return T2IPipe()

    @cached_property
    def node_tuner(self):
        from tuner import NodeTuner
        return NodeTuner()

    @cached_property
    def sys_cap(self):
        from capacity import SystemCapacity
        return SystemCapacity()
