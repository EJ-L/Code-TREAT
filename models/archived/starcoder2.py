from HuggingFaceBase import HuggingFaceBase
from APIBase import APIBase

# check models: https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a
# use examples: https://github.com/bigcode-project/starcoder2


class StarCoder2(HuggingFaceBase):
    def __init__(self, api_key, model_size, purpose='text-generation'):
        available_sizes = ["3b", "7b", "15b"]
        if model_size not in available_sizes:
            raise ValueError(f"Model size must be one of {available_sizes}")
        model_name = f"bigcode/starcoder2-{model_size}"
        super().__init__(api_key, model_name, purpose)
        self.prompt = ""
        self.setup_model()
        self.setup_tokenizer()
        

# export REPLICATE_API_TOKEN=<paste-your-token-here>
class StarCoder2_15B(APIBase):
    def __init__(self, api_key, base_url, client_library):
        model_name = f""
        self.api_key = api_key
        self.base_url = base_url
        self.client_library = client_library
        super.__init__(api_key, base_url, model_name)
        self.setup_client(client_library)
        self.text_messages = ""
        
    # check: https://replicate.com/cjwbw/starcoder2-15b/examples?input=python
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(self, messages, temperature=temperature, max_tokens=max_tokens, delay=delay_time, top_k=top_k, top_p=top_p):
        time.sleep(delay)
        
        input = {
            "prompt": messages,
            "temperature": temperature,
            "max_new_tokens": max_tokens
        }
        
        if top_k is not None:
            input["top_k"] = top_k

        if top_p is not None:
            input["top_p"] = top_p

        response = self.client(
            "cjwbw/starcoder2-15b:d67b7d32b63bb8a2cf6b95c523921408e38ce7d7228fdff7b1eb636dc2c5ecd8",
            input=input
        )

        return response