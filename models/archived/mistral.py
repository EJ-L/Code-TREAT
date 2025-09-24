from Framework.Models.APIBase import APIBase
class Mistral(APIBase):
    def __init__(self, api_key, base_url, model_name, client):
        # main testing models: "open-mixtral-8x7b", "open-mixtral-8x22b", "mistral-large-2407", "mistral-small-2409"
        model_names = ["open-mixtral-8x7b", "open-mixtral-8x22b", "mistral-large-2407", "mistral-small-2409", "codestral-2405", "open-mistral-7b", "mistral-small-latest", "mistral-medium-latest", "open-mistral-nemo-2407", "pixtral-12b-2409", "mistral-embed"]
        if model_name not in model_names:
            raise ValueError("Invalid model")
        super().__init__(api_key, base_url, model_name)
        self.setup_client(client)