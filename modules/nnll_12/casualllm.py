from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "darkshapes/suzume-llama-3-8B-multilingual-orpo-borda-top25-gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a senior level programmer who gives an accurate and concise examples within the scope of your knowledge, while disclosing when a request goes beyond it."},
    {"role": "user", "content": "Return python code to access a webcam with as few dependencies as possible."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.0,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

