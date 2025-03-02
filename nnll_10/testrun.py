import os
import dspy

class ChatWithListMemory(dspy.Module):
    def __init__(self, memory_size=5):
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
        self.chat_instance = dspy.Predict("message, history -> answer")

    def forward(self, message: str):
        self.memory.append(message)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        combined_context = " ".join(self.memory)
        response = self.chat_instance(message=message, history=combined_context)
        return response

model = 'ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M'

local_llama = dspy.LM(api_base='http://localhost:11434/api/chat', model=model,  model_type='chat')
dspy.settings.configure(lm=local_llama)
chat_module = ChatWithListMemory()

def chat(prompt_input):
    print(f"{model} Loaded. System ready.")
    while True:
        user_input = prompt_input# input("User: ") # replace with variable from __init__
        if user_input.lower() == "exit":
            print("---")
            break
        response = chat_module(message=user_input)
        return (f"{os.path.basename(model)}: ", response.completions[0].answer)



# lm = dspy.
# dspy.configure(lm=lm)
# with dspy.context(lm=lm):
#     lm("Hello, just testing")
#     lm(messages=[{"role": "user", "content": "Say this is a test!"}])

# import dspy

# async def main():
#     client = ollama.AsyncClient()
#     for part in await client.generate(sys.args[0], sys.args[1], stream=True):
#         print(part['response'], end='', flush=True)

# async def send_request():
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print('\nGoodbye!')
