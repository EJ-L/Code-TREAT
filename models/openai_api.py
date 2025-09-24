from .providers import ProviderFactory

class OpenAI:
    def __init__(self, api_key, base_url, model_name, client):
        self._api = ProviderFactory.create("openai", api_key, base_url, model_name, client)

    # delegate methods used by call sites
    def chat(self, *args, **kwargs):
        return self._api.chat(*args, **kwargs)