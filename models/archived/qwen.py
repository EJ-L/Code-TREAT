from models.apibase import APIBase
class Qwen(APIBase):
    def __init__(self, api_key, base_url, model_name, client):
        if base_url.find("deepinfra") != -1:
            model_names = ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-72B-Instruct"]
            if model_name not in model_names:
                raise ValueError("Invalid model size")
        if base_url.find("aliyun") != -1:
            model_names = model_names = ["qwen-long","qwen-max", "qwen-max-0428", "qwen-max-0403","qwen-max-0107", "qwen-max-longcontext","qwen-plus", "qwen-plus-0806", "qwen-plus-0723","qwen-plus-0624", "qwen-plus-0206", "qwen-turbo", "qwen-turbo-0624", "qwen-turbo-0206", "qwen2-57b-a14b-instruct", "qwen2-72b-instruct", "qwen2-7b-instruct", "qwen2-1.5b-instruct", "qwen2-0.5b-instruct", "qwen1.5-110b-chat", "qwen1.5-72b-chat", "qwen1.5-32b-chat", "qwen1.5-14b-chat", "qwen1.5-7b-chat", "qwen1.5-1.8b-chat", "qwen1.5-0.5b-chat", "codeqwen1.5-7b-chat", "qwen-72b-chat", "qwen-14b-chat", "qwen-7b-chat", "qwen-1.8b-longcontext-chat", "qwen-1.8b-chat", "qwen2-math-72b-instruct", "qwen2-math-7b-instruct", "qwen2-math-1.5b-instruct"]
            if model_name not in model_names:
                raise ValueError("Invalid model size")
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)