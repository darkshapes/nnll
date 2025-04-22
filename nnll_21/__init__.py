from enum import Enum
from typing import Literal


class AspectImage(str, Enum):
    """Aspects for 2D image models
    incl. Flux, SD3, SDXL, AuraFlow"""

    RATIOS = {
        "1:1___1024x1024": (1024, 1024),
        "16:15_1024x960": (1024, 960),
        "17:15_1088x960": (1088, 960),
        "17:14_1088x896": (1088, 896),
        "18:13_1152x832": (1152, 832),
        "4:3___1152x896": (1152, 896),
        "3:2___1216x832": (1216, 832),
        # "72:32_1232x832" : ( 1232, 832),
        "5:3___1280x768": (1280, 768),
        "21:11_1344x704": (1344, 704),
        "7:4___1344x768": (1344, 768),
        "2:1___1408x704": (1408, 704),
        "23:11_1472x704": (1472, 704),
        "21:9__1536x640": (1536, 640),
        "2:1___1536x768": (1536, 768),
        "5:2___1600x640": (1600, 640),
        "26:9__1664x576": (1664, 576),
        "3:1___1728x576": (1728, 576),
        "28:9__1792x576": (1792, 576),
        "29:8__1856x512": (1856, 512),
        "15:4__1920x512": (1920, 512),
        "31:8__1984x512": (1984, 512),
        "4:1___2048x512": (2048, 512),
    }


class AspectVideo(str, Enum):
    """Aspects for Video models
    incl. HunyuanVideo, Pyramid, Sora"""

    RATIOS = {
        "1:1___V_256x256": (256, 256),
        "4:3___V_320x240": (320, 240),
        "32:27_V_576x486": (576, 486),
        "22:15_V_704x480": (704, 480),
        "9:5___V_720x400": (720, 400),
        "3:2___V_720x480": (720, 480),
        "5:4___V_720x576": (720, 576),
        "3:2___V_768x512": (768, 512),
        "4:3___V_832x624": (832, 624),
        "53:30_V_848x480": (848, 480),
        "4:3___V 960x704": (960, 704),
        "1:1___V_960x960": (960, 960),
        "20:11_V_1280x704": (1280, 704),
        "16:9__V_1024X576": (1024, 576),
    }


class AspectRender(str, Enum):
    """Aspects for 3d-generative models
    incl. SV3D"""

    RATIOS = {
        "1:1__SV3D_576x576": (576, 576),
    }


class AspectLegacy(str, Enum):
    """Aspect ratios for earlier 2d Diffusion models
    incl. Latent/Stable Diffusion, Pixart A, Playground 1, etc"""

    RATIOS = {
        "1:1____SD_512x512": (512, 512),
        "4:3____SD_682x512": (682, 512),
        "3:2____SD_768x512": (768, 512),
        "1:1____SD_768x768": (768, 768),
        "16:9___SD_910x512": (910, 512),
        "1:85:1_SD_952x512": (952, 512),
        "2:1____SD_1024x512": (1024, 512),
    }


class Precision(str, Enum):
    MIXED = "mixed"
    FP64 = "float64"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8E4M3FN = "float8_e4m3fn"
    FP8E5M2 = "float8_e5m2"
    IN64 = "int64"
    IN32 = "int32"
    IN16 = "int16"
    IN8 = "int8"
    UN8 = "uint8"
    NF4 = "nf4"


class TensorDataType:
    TYPE_T = Literal["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "nf4", "BOOL"]
    TYPE_R = Literal["fp64", "fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "i64", "i32", "i16", "i8", "u8", "nf4", "bool"]


# class TensorData:
#     dtype: DTYPE_T
#     shape: List[int]
#     data_offsets: Tuple[int, int]
#     parameter_count: int = field(init=False)

#     def __post_init__(self) -> None:
#         # Taken from https://stackoverflow.com/a/13840436
#         try:
#             self.parameter_count = functools.reduce(operator.mul, self.shape)
#         except TypeError:
#             self.parameter_count = 1  # scalar value has no shape
