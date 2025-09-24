
from models.apibase import APIBase
# https://open.bigmodel.cn/dev/api/code-model/codegeex-4
class Gemma2(APIBase):
    def __init__(self, api_key, base_url, model_size, client):
        # we aim to test only 9b
        if model_size not in ["9b", "27b"]:
            raise ValueError("Invalid model size. Please choose from '9b' or '27b'.")
        
        model_name = f"google/gemma-2-{model_size}-it"
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)
        