import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig, AutoModel, pipeline
from Framework.utils import *
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import time
class HuggingFaceBase:
    def __init__(self, api_key_env, model_name, purpose="text-generation"):
        purpose_list = ["question-answering", "summarization", "text-classification", "text-generation", "token-classification", "automatic-speech-recognition", "audio-classification", "object-detection", "image-segmentation", "reinforcement-learning"]
        if purpose not in purpose_list:
            raise ValueError(f"Invalid purpose. Please choose from {purpose_list}")
        self.api_key = os.getenv(api_key_env)
        self.model_name = model_name
        # set up checking for purpose
        self.purpose = purpose
        
    # check: https://github.com/bigcode-project/starcoder2
    # torch_dtype=torch.float16, bit_precision=4 or 8
    def setup_model(self, model_function, gpu=True, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = device.type == "cuda" and gpu 
        
        self.model = model_function(self.model_name, **kwargs)

    def setup_tokenizer(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer(self.model_name, **kwargs)
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(self, messages, n=n, max_tokens=max_tokens, delay=delay_time):
        time.sleep(delay)
        # tokenize the messages
        model_inputs = self.tokenizer([messages], return_tensors="pt").to(self.model.device)
        # generate the response
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=False)
        # decode the response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # or 
        # pipe = pipeline(self.purpose, model=self.model, tokenizer=self.tokenizer, device="auto")
        # response = pipe(messages)
        if n == 1:
            return response[0]
        else:
            return response
        