import litellm
import huggingface_hub

litellm.disable_end_user_cost_tracking
litellm.disable_hf_tokenizer_download
litellm.telemetry = False
huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY = True
# LITELLM_LOCAL_MODEL_COST_MAP="True"
