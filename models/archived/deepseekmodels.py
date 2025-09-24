"""
base_url for MAX_OUTPUT_TOKENS:
4K: https://api.deepseek.com, 
8K:"https://api.deepseek.com/beta"
"""
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from HuggingFaceBase import HuggingFaceBase
from models.apibase import APIBase
from models.apibase import APIBase
# https://open.bigmodel.cn/dev/api/code-model/codegeex-4
class DeepSeekAPI(APIBase):
    def __init__(self, api_key, base_url, model_version, client):       
        if model_version.lower() == 'v3':
            model_name = f"deepseek-chat"
        if model_version.lower() == 'r1':        
            model_name = f"deepseek-reasoner" 
        # model_name = f"deepseek-ai/DeepSeek-{model_version}"
        super().__init__(api_key, base_url, model_name)
        
        self.setup_client(client)

# class DeepSeekAPI(APIBase):
#     """
#     The Official Deepseek API
#     """
#     def __init__(self, api_key, base_url, model_version, client, system_prompt="You are a helpful asssitant"):
#         # set the model name
#         if model_version.lower() == 'v3':
#             model_name = f"deepseek-chat"
#         if model_version.lower() == 'r1':        
#             model_name = f"deepseek-reasoner"
#         # call the parent class constructor
#         super().__init__(api_key, base_url, model_name)
#         self.text_based_messages =[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             }
#         ]

#         # self.image_based_messages = None
#         self.setup_client(client) 
        
# https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base
# https://huggingface.co/deepseek-ai/DeepSeek-V2 : slight modification needed
# class DeepSeekModels(HuggingFaceBase):
#     def __init__(self, api_key, model_version, purpose='text-generation'):
#         versions = ["Coder-V2-Instruct", "Coder-V2-Base", "Coder-V2-Lite", "Coder-V2-Lite-Base", 'V2-Chat', 'V2', 'V2-Lite', 'V2-Lite-Chat']
#         if model_version not in versions:
#             raise ValueError(f"Invalid model version. Choose from {versions}")
#         model_name = f"deepseek-ai/DeepSeek-{model_version}"
#         super().__init__(api_key, model_name, purpose)
#         self.setup_model(AutoModelForCausalLM.from_pretrained, 
#                        gpu=True, 
#                        torch_dtype=torch.bfloat16, 
#                        trust_remote_code=True)
#         self.setup_tokenizer(AutoTokenizer.from_pretrained,
#                            trust_remote_code=True)
        
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # def code_completion(self, messages, n=n, max_tokens=max_tokens, delay=delay_time):
    #     time.sleep(delay)
    #     inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
    #     outputs = self.model.generate(**inputs, max_length=max_tokens)
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return response
   
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # def code_insertion(self, messages, n=n, max_tokens=max_tokens, delay=delay_time):
    #     time.sleep(delay)
    #     inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
    #     outputs = self.model.generate(**inputs, max_length=max_tokens)
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(messages):]
    #     return response
             
        
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    # def chat_completion(self, messages, n=n, max_tokens=max_tokens, delay=delay_time):
    #     time.sleep(delay)
    #     inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    #     outputs = self.model.generate(inputs, max_new_tokens=max_tokens, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
    #     response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    #     return response