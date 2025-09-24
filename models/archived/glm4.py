from models.apibase import APIBase
# https://open.bigmodel.cn/dev/api/normal-model/glm-4
class GLM4(APIBase):
    def __init__(self, api_key, base_url, model_name, client):
        available_models = ["glm-4-plus", "glm-4-0520", "glm-4", "glm-4-air", "glm-4-airx", "glm-4-long", "glm-4-flash", "glm-4.5-flash"]
        if model_name not in available_models:
            raise ValueError(f"Invalid model name. Available models are {available_models}")
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)
        
        # need to add     
        # stop=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>"]