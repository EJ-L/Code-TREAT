from models.apibase import APIBase

class Llama_Instruct3(APIBase):
    def __init__(self, api_key, base_url, model_size, client):
        model_sizes = ['8B', '70B']
        if model_size not in model_sizes:
            raise ValueError("model_size must be either '8B' or '70B'")
        model_name = f"meta-llama/Meta-Llama-3.1-{model_size}-Instruct"
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)