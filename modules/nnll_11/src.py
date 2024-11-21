
from transformers import (
    CLIPTokenizer,
    CLIPTokenizerFast,
    T5Tokenizer,
    T5TokenizerFast,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    AutoModel
)
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    FluxPipeline,
    AuraFlowPipeline,
    AutoencoderKL,
    AutoencoderTiny,
    StableCascadePriorPipeline,
    StableCascadeDecoderPipeline,
    StableCascadeCombinedPipeline
)

from diffusers.schedulers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    EDMDPMSolverMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LCMScheduler,
    TCDScheduler,
    AysSchedules,
    HeunDiscreteScheduler,
    UniPCMultistepScheduler,
    LMSDiscreteScheduler,
    DEISMultistepScheduler,
    DDPMWuerstchenScheduler
)

tokenizer_classes = {
    "CLIPTOKENIZER"    : CLIPTokenizer,     # CLIP L
    "CLIPTOKENIZERFAST": CLIPTokenizerFast, # CLIP G
    "T5TOKENIZERFAST"  : T5TokenizerFast,   # T5 XXL
    "T5TOKENIZER"      : T5Tokenizer,       # T5 XXL
    "AUTOTOKENIZER"    : AutoTokenizer,     # CHATGLM
}

encoder_classes = {
    "CLIPTEXTMODEL"              : CLIPTextModel,               # CLIP L
    "CLIPTEXTMODELWITHPROJECTION": CLIPTextModelWithProjection, # CLIP G
    "T5ENCODERMODEL"             : T5EncoderModel,              # T5
    "AUTOMODEL"                  : AutoModel,                   # CHATGLM
}

autoencoder_classes = {
   "AUTOKL": AutoencoderKL,
   "AUTOTNY": AutoencoderTiny
}

pipe_classes = {
    "AUTOPIPE"       : AutoPipelineForText2Image,
    "AURAFLOWPIPE"   : AuraFlowPipeline,
    "FLUXPIPE"       : FluxPipeline,
    "SDXLPIPE"       : StableDiffusionXLPipeline,
    "SD15PIPE"       : StableDiffusionPipeline,
    "CASCADEPRIOR"   : StableCascadePriorPipeline,
    "CASCADEDECODER" : StableCascadeDecoderPipeline,
    "CASCADECOMBINED": StableCascadeCombinedPipeline

}

scheduler_classes = {
    "EULERDISCRETE"         : EulerDiscreteScheduler,
    "EULERANCESTRAL"        : EulerAncestralDiscreteScheduler,
    "FLOWMATCHEULERDISCRETE": FlowMatchEulerDiscreteScheduler,
    "EPMDPMSOLVER"          : EDMDPMSolverMultistepScheduler,
    "DPMSOLVER"             : DPMSolverMultistepScheduler,
    "DDIM"                  : DDIMScheduler,
    "LCM"                   : LCMScheduler,
    "TCD"                   : TCDScheduler,
    "AYS"                   : AysSchedules,
    "HEUNDISCRETE"          : HeunDiscreteScheduler,
    "UNIPCMULTISTEP"        : UniPCMultistepScheduler,
    "LMSDISCRETE"           : LMSDiscreteScheduler,
    "DEISMULTISTEP"         : DEISMultistepScheduler,
    "DDPMWUERSTCHEN"        : DDPMWuerstchenScheduler
}

def method_crafter(class_name: dict, method_name:str, location:str, expressions:dict):
    """
    #### Facilitates dynamic and iterative creation of 🧨 Diffusers and 🤗 Transformers classes.
    #### `key_class`  : *`_classes` [`scheduler`/`tokenizer`/`pipeline`] a key from a dict of known library classes
    #### `method_name`: `from_`* [`config`/`single_file`/`pretrained`] the desired class method to load with
    #### `location`   : path to the model or configuration appropriate for the method
    #### `expressions`: a set of arguments to pass to the method
    #### OUTPUT       : a `dict` containing the formatted request to instantiate the class
    """
    config_methods = {"from_config", "from_single_file", "from_pretrained",}
    if method_name not in config_methods:
        raise AttributeError(f"Method {method_name} not found")
    else:
        return {
            key: getattr(cls, method_name)(location, **expressions)
            for key, cls in class_name.items()
        }