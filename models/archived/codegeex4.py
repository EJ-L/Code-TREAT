from Framework.Models.APIBase import APIBase
# https://open.bigmodel.cn/dev/api/code-model/codegeex-4
class CodeGeeX4(APIBase):
    def __init__(self, api_key, base_url, client):
        model_name = "codegeex-4"
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)
        
        # need to add     
        # stop=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>"]