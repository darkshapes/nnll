import litellm
import huggingface_hub

litellm.disable_end_user_cost_tracking = True
litellm.disable_hf_tokenizer_download = True
litellm.telemetry = False
huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY = True
# huggingface_hub.constants.HF_HUB_VERBOSITY
