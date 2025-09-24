from CodeLLaMa_Ins import CodeLLaMa_Ins
from DeepSeekV2_5 import DeepSeekV2_5

api_key = "AIML_KEY"
base_url = "https://api.aimlapi.com"
model_size = "7B"
client_library = "OPENAI_CLIENT"
# Test case for CodeLLaMA-Ins
model = CodeLlaMa_Ins(api_key, base_url, model_size, client_library)
model.chat(messages="def Fib(n):")

api_key = "DEEPSEEK_KEY"
base_url = "https://api.aimlapi.com"
client_library = "OPENAI_CLIENT"
# Test case for CodeLLaMA-Ins
model = DeepSeekV2_5(api_key, base_url, client_library)
model.chat(messages="def Fib(n):")


