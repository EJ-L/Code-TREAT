from .providers import ProviderFactory

class ChatAnywhereAPI:
    def __init__(self, api_key, base_url, model_name, client):
        self._api = ProviderFactory.create("chatanywhere", api_key, base_url, model_name, client)

    def chat(self, *args, **kwargs):
        return self._api.chat(*args, **kwargs)