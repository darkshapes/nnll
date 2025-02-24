from llama_cpp import Llama
from trndup import softRndmc

seed = softRndmc.softRandom()
print(seed)

llm = Llama(
      model_path="/Users/Shared/ouerve/recent/darkshapes/models/llms/codeninja-1.0-openchat-7b.Q5_K_M.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      n_threads=8,   # The number of CPU threads to use, tailor to your system and the resulting performance
      seed=seed, # Uncomment to set a specific seed
      n_ctx=8192, # Uncomment to increase the context window
      chat_format="openchat",
      repeat_penalty=1,
      temperature=0,
)

response = llm.create_chat_completion(
    messages=[
        { "role": "system", "content": "You are a senior level programmer who gives an accurate and concise examples within the scope of your knowledge, while disclosing when a request goes beyond it." },
        {
            "role": "user",
            "content": "Return python code to access a webcam with as few dependencies as possible."
        }
    ],
    stream=True
)

for chunk in response:
    delta = chunk['choices'][0]['delta']
    if 'role' in delta:
        print(delta['role'], end=': ')
    elif 'content' in delta:
        print(delta['content'], end='')