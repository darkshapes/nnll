# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


"""確認 Data Type Validation"""

from ast import Constant
from platform import python_version_tuple

from pydantic import AfterValidator, BaseModel, Field, TypeAdapter, field_validator
from pydantic_core import ValidationError

if float(python_version_tuple()[0]) == 3.0 and float(python_version_tuple()[1]) <= 12.0:
    from typing_extensions import Annotated, List, Set, TypedDict, Union

else:
    from typing import Annotated, List, Set, TypedDict, Union


class EmptyField:
    "When no data is available"

    PLACEHOLDER: str = "No Data"
    EMPTY: str = "Empty"
    LABELS: List[Constant] = [PLACEHOLDER, EMPTY]


class UpField:
    """Upper display area for ui\n
    Tags in Up or Down arrange data layout for UI
    Ensure ALL field tags are included in LABELS
    """

    METADATA: str = "Metadata"
    PROMPT: str = "Prompt Data"
    TAGS: str = "Tags"
    TEXT_DATA: str = "TEXT Data"
    DATA: str = "DATA"
    LABELS: List[Constant] = [
        METADATA,
        PROMPT,
        TAGS,
        TEXT_DATA,
        DATA,
    ]


class DownField:
    """Lower display area for ui\n
    Tags in Up or Down arrange data layout for UI
    Ensure ALL field tags are included in LABELS

    """

    GENERATION_DATA: str = "Generation Data"
    SYSTEM: str = "System"
    ICC: str = "ICC Profile"
    EXIF: str = "EXIF"
    RAW_DATA: str = "Text Data"
    LAYER_DATA: str = "Layer Data"
    JSON_DATA: str = "JSON Data"
    TOML_DATA: str = "TOML Data"
    LABELS: List[Constant] = [
        GENERATION_DATA,
        SYSTEM,
        ICC,
        EXIF,
        RAW_DATA,
        LAYER_DATA,
        TOML_DATA,
        JSON_DATA,
    ]


class ExtensionType:
    """Valid file formats for metadata reading\n"""

    PNG_: Set[str] = {".png"}
    JPEG: Set[str] = {".jpg", ".jpeg"}
    WEBP: Set[str] = {".webp"}
    JSON: Set[str] = {".json"}
    TOML: Set[str] = {".toml"}
    TEXT: Set[str] = {".txt", ".text"}
    HTML: Set[str] = {".html", ".htm"}
    XML_: Set[str] = {".xml"}
    GGUF: Set[str] = {".gguf"}
    SAFE: Set[str] = {".safetensors", ".sft"}
    PICK: Set[str] = {".pt", ".pth", ".ckpt", ".pickletensor"}
    ONNX: Set[str] = {".onnx"}

    IMAGE: List[str] = list(JPEG.union(WEBP, PNG_))
    EXIF: List[str] = list(JPEG.union(WEBP))
    SCHEMA: List[str] = list(JSON.union(TOML))
    PLAIN: List[str] = list(TEXT.union(XML_, HTML))
    MODEL: List[str] = list(SAFE.union(GGUF, PICK, ONNX))

    MEDIA: List[str] = IMAGE + EXIF + SCHEMA + PLAIN

    IGNORE: List[Constant] = [
        "Thumbs.db",
        "desktop.ini",
        ".fseventsd",
        ".DS_Store",
        ".gitattributes",
        ".env",
        ".py",
        "LICENSE",
        ".md",
    ]


class NodeNames:
    """Node names that carry prompts inside"""

    ENCODERS = {
        "CLIPTextEncodeFlux",
        "CLIPTextEncodeSD3",
        "CLIPTextEncodeSDXL",
        "CLIPTextEncodeHunyuanDiT",
        "CLIPTextEncodePixArtAlpha",
        "CLIPTextEncodeSDXLRefiner",
        "ImpactWildcardEncodeCLIPTextEncode",
        "BNK_CLIPTextEncodeAdvanced",
        "BNK_CLIPTextEncodeSDXLAdvanced",
        "WildcardEncode //Inspire",
        "TSC_EfficientLoader",
        "TSC_EfficientLoaderSDXL",
        "RgthreePowerPrompt",
        "RgthreePowerPromptSimple",
        "RgthreeSDXLPowerPromptPositive",
        "RgthreeSDXLPowerPromptSimple",
        "AdvancedCLIPTextEncode",
        "AdvancedCLIPTextEncodeWithBreak",
        "Text2Prompt",
        "smZ CLIPTextEncode",
        "CLIPTextEncode",
    }
    STRING_INPUT = {
        "RecourseStrings",
        "StringSelector",
        "ImpactWildcardProcessor",
        "CText",
        "CTextML",
        "CListString",
        "CSwitchString",
        "CR_PromptText",
        "StringLiteral",
        "CR_CombinePromptSDParameterGenerator",
        "WidgetToString",
        "Show Text 🐍",
    }
    PROMPT_LABELS = ["Positive prompt", "Negative prompt", "Prompt"]

    IGNORE_KEYS = [
        "type",
        "link",
        "shape",
        "id",
        "pos",
        "size",
        "node_id",
        "empty_padding",
    ]

    DATA_KEYS = {
        "class_type": "inputs",
        "nodes": "widget_values",
    }
    PROMPT_NODE_FIELDS = {
        "text",
        "t5xxl",
        "clip-l",
        "clip-g",
        "mt5",
        "mt5xl",
        "bert",
        "clip-h",
        "wildcard",
        "string",
        "positive",
        "negative",
        "text_g",
        "text_l",
        "wildcard_text",
        "populated_text",
    }


# EXC_INFO: bool = LOG_LEVEL != "i"


def bracket_check(maybe_brackets: str | dict):
    """
    Check and correct brackets in a kv pair format string\n
    :param maybe_brackets: The data that may or may not have brackets
    :type maybe_brackets: `str` | `dict`
    :return: the corrected string, or a dict
    """
    if isinstance(maybe_brackets, dict):
        pass
    elif isinstance(maybe_brackets, str):
        if next(iter(maybe_brackets)) != "{":
            maybe_brackets = "{" + maybe_brackets
        if maybe_brackets[-1:] != "}":
            maybe_brackets += "}"
    else:
        raise ValidationError("Check input must be str or dict")
    return maybe_brackets


class NodeDataMap(TypedDict):
    """Valid nodeui json prompt structure"""

    class_type: str
    inputs: Union[dict, float]


class NodeWorkflow(TypedDict):
    """Valid nodeui json workflow structure"""

    last_node_id: int
    last_link_id: Union[int, dict]
    nodes: list
    links: list
    groups: list
    config: dict
    extra: dict
    version: float


class BracketedDict(BaseModel):
    """
    Ensure a string value is formatted correctly for a dictionary\n
    :param node_data: k/v pairs with or without brackets
    :type node_data: `str`, required
    """

    bracketed: Annotated[str, Field(init=False, exclude=True), AfterValidator(bracket_check)]


class IsThisNode:
    """
    Confirm the data input of a ComfyUI dict\n
    :param node_data: The data to verify
    :type node_data: `str | dict`
    """

    data = TypeAdapter(NodeDataMap)
    workflow = TypeAdapter(NodeWorkflow)


class ListOfDelineatedStr(BaseModel):
    """Ensure list conversion into delineated string\n"""

    convert: list

    @field_validator("convert")
    @classmethod
    def drop_tuple(cls, regex_match: list):
        """Remove tuple elements from validation"""
        regex_match = list(next(iter(regex_match), None))
        return regex_match


BREAKING_SUFFIX = r".*(?:-)(prior)$|.*(?:-)(diffusers)$|.*[_-](\d{3,4}px|-T2V$|-I2V$)"
PARAMETERS_SUFFIX = r"(\d{1,4}[KkMmBb]|[._-]\d+[\._-]\d+[Bb][._-]).*?$"
SEARCH_SUFFIX = r"\d+[._-]?\d+[BbMmKk](it)?|[._-]\d+[BbMmKk](it)?"
