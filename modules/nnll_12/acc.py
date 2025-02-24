import torch
import trndup
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM
from trndup import softRandom

model_id = "/Users/Shared/ouerve/recent/darkshapes/models/llms/codeninja-1.0-openchat-7b.Q5_K_M.gguf"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

seed = softRandom()
set_seed(seed)
export_from_model(model, output="ov_model", task="text-generation-with-past")

input = torch.randn(2,3)
input = input.to("cpu")
output = model(input)

outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))