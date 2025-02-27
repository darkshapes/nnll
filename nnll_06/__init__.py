### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from mlx_lm import load, generate


def main():
    print("Hello from mlx!")

    model, tokenizer = load("RefalMachine/RuadaptQwen2.5-32B-Pro-Beta", tokenizer_config={"eos_token": "<|endoftext|>"})  "model_max_length": 131072,
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You answer questions about code, and always use less than 100 words to explain it. Predictions with low success-probability you approach with a step-by-step method, solving challenges incrementally on the way to a total solution."

    prompt = "hello"
"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. You answer questions about code, and always use less than 100 words to explain it. Predictions with low success-probability you approach with a step-by-step method, solving challenges incrementally on the way to a total solution.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate(model, tokenizer, prompt=prompt, verbose=True)


# PARAMETER num_ctx 32768
def chatbot():
    prompt = "hello"
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    return data


if __name__ == "__main__":
    main()

# mlx_lm.server --port 2097 --model RefalMachine/RuadaptQwen2.5-32B-Pro-Beta --chat-template "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. You answer questions about code, and always use less than 100 words to explain it. Predictions with low success-probability you approach with a step-by-step method, solving challenges incrementally on the way to a total solution.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

curl localhost:2097/v1/chat/completions -H "Content-Type: application/json" -d '{ "messages": [{"role": "user", "content": "Demonstrate a Python function that retains the path to its source module location."}], }'


curl localhost:2097/v1/chat/completions -H "Content-Type: application/json" -d '{ "messages": [{"role": "user", "content": "Demonstrate a Python function that retains the path to its source module location."}] }'