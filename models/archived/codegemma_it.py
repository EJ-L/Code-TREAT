from transformers import GemmaTokenizer, AutoModelForCausalLM
from Framework.Models.HuggingFaceBase import HuggingFaceBase
import torch
from Framework.utils import *
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import time
# https://huggingface.co/google/codegemma-7b-it
class CodeGemma_it(HuggingFaceBase):
    def __init__(self, api_key, purpose='text-generation'):
        model_name = 'google/codegemma-7b-it'
        super().__init__(api_key, model_name, purpose)
        '''
        Code Geneartion:
        "Write me a Python function to calculate the nth fibonacci number."
        '''
        self.setup_model(GemmaTokenizer.from_pretrained)
        self.setup_tokenizer(AutoModelForCausalLM.from_pretrained)
        self.text_message = "" 
        '''
        Chat:
        [
            { "role": "user", "content": "Write a hello world program" },
        ]
        '''
        self.setup_model(GemmaTokenizer.from_pretrained)
        self.setup_tokenizer(AutoModelForCausalLM.from_pretrained,
                            gpu=True,
                            device_map="cuda",
                            torch_dtype="torch.bfloat16")
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat_completion(self, messages, n=n, max_tokens=max_tokens, delay=delay_time):
        time.sleep(delay)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=max_tokens)
