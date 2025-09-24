from models.apibase import APIBase

class Qwen2(APIBase):
    def __init__(self, api_key, base_url, model_size, client):
        model_sizes = ["7B", "72B"]
        if model_size not in model_sizes:
            raise ValueError("Invalid model size")
        if base_url.find("deepinfra") != -1:
            model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"
        if base_url.find("aliyun") != -1:
            model_name = "qwen2-" + model_size + "-instruct"
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)