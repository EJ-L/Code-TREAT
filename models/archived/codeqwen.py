from Framework.Models.APIBase import APIBase
# https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes
class CodeQwen1_5_Chat(APIBase):
    def __init__(self, api_key, base_url, client):
        model_name = "codeqwen1.5-7b-chat"
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)