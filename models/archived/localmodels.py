from models.apibase import APIBase
from openai import OpenAI
class Ollama(APIBase):
    def __init__(self, api_key, base_url, model, client):
        super().__init__(api_key, base_url, model)
        self.setup_client(client)
        