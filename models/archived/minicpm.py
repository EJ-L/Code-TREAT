from models.apibase import APIBase
# https://open.bigmodel.cn/dev/api/code-model/codegeex-4
class Minicpm(APIBase):
    def __init__(self, api_key, base_url, client):
        model_name = "openbmb/MiniCPM-Llama3-V-2_5"
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)
        