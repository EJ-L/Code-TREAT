from __future__ import annotations
from dotenv import load_dotenv
import os

from .providers import ProviderFactory
from .apibase import APIBase

load_dotenv('.env')

# --- ENV KEYS ---
DEEPINFRA_KEY = os.environ.get('DEEPINFRA_KEY')
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_KEY')
CHATANYWHERE_KEY = os.environ.get('CHATANYWHERE_KEY')
GROK_KEY = os.environ.get('GROK_KEY')
OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY')
ZHIPU_KEY = os.environ.get('ZHIPU_KEY')

# --- BASE URLS ---
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
CHATANYWHERE_URL = "https://api.chatanywhere.tech/v1"
GROK_URL = "https://api.x.ai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# --- CLIENT IDS (these are names consumed by APIBase.setup_client) ---
DEEPINFRA_CLIENT = "DEEPINFRA_CLIENT"
DEEPSEEK_CLIENT = "DEEPSEEK_CLIENT"
OPENROUTER_CLIENT = "OPENROUTER_CLIENT"
OPENAI_CLIENT = "OPENAI_CLIENT"
GROK_CLIENT = "GROK_CLIENT"
ZHIPU_CLIENT = "ZHIPU_CLIENT"
OLLAMA_CLIENT = "OLLAMA"
CHATANYWHERE_CLIENT = "CHATANYWHERE_CLIENT"

def mk(provider: str, api_key: str, base_url: str, model_name: str, client: str):
    """Helper to reduce boilerplate when constructing from ProviderFactory.
    - provider: one of the registered providers (openai, openrouter, deepinfra, deepseek, chatanywhere, zhipu, ollama, ...)
    - client: optional override that gets passed to APIBase.setup_client (keeps your current per-entry behavior)
    """
    return ProviderFactory.create(provider, api_key, base_url, model_name, client)

def mk_direct_api(api_key: str, base_url: str, model_name: str, client: str):
    """Direct APIBase creation bypassing factory for debugging"""
    api = APIBase(api_key_env=api_key, base_url=base_url, api_model_name=model_name)
    api.setup_client(client)
    return api


# NOTE: If duplicate keys appear in this dict, Python keeps the last one.
# I preserved your keys as-is for minimal change. Consider deduping later.
MODELS = {
    # x.ai / Grok (OpenAI-compatible)
    "grok-3-mini": mk("openai", GROK_KEY, GROK_URL, "grok-3-mini", GROK_CLIENT),
    # DeepInfra-hosted models
    "gemma-2-9b-it": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "gemma-2-9b-it", DEEPINFRA_CLIENT),
    "gemma-2-27b-it": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "gemma-2-27b-it", DEEPINFRA_CLIENT),
    "llama-3.1-8B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "meta-llama/Meta-Llama-3.1-8B-Instruct", DEEPINFRA_CLIENT),
    "llama-3.1-70B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "meta-llama/Meta-Llama-3.1-70B-Instruct", DEEPINFRA_CLIENT),
    "llama-4-scout": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "meta-llama/Llama-4-Scout-17B-16E-Instruct", DEEPINFRA_CLIENT),
    "Qwen2.5-72B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "Qwen/Qwen2.5-72B-Instruct", DEEPINFRA_CLIENT),
    "QwQ-32B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "Qwen/QwQ-32B", DEEPINFRA_CLIENT),
    "gemma-3-27b-it": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "google/gemma-3-27b-it", DEEPINFRA_CLIENT),
    "Qwen2.5-Coder-32B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "Qwen/Qwen2.5-Coder-32B-Instruct", DEEPINFRA_CLIENT),
    "llama-3.3-70B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "meta-llama/Llama-3.3-70B-Instruct", DEEPINFRA_CLIENT),
    "Qwen3-30B-A3B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "Qwen/Qwen3-30B-A3B", DEEPINFRA_CLIENT),
    "Qwen3-32B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "Qwen/Qwen3-32B", DEEPINFRA_CLIENT),
    "Qwen3-235B-A22B": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "Qwen/Qwen3-235B-A22B", DEEPINFRA_CLIENT),
    "deepseek-v3": mk("deepinfra", DEEPINFRA_KEY, DEEPINFRA_URL, "deepseek-ai/DeepSeek-V3", DEEPSEEK_CLIENT),
    # DeepSeek (native)
    # "deepseek-v3": mk("deepseek", DEEPSEEK_KEY, DEEPSEEK_URL, "deepseek-ai/DeepSeek-V3", DEEPSEEK_CLIENT),
    # "deepseek-r1": mk("deepseek", DEEPSEEK_KEY, DEEPSEEK_URL, "deepseek-ai/DeepSeek-R1", DEEPSEEK_CLIENT),
    # "deepseek-r1-0528": mk("deepseek", DEEPSEEK_KEY, DEEPSEEK_URL, "deepseek-ai/DeepSeek-R1-0528", DEEPSEEK_CLIENT),

    # ChatAnywhere (OpenAI-compatible umbrella)
    "gpt-4.1-2025-04-14": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-4.1-2025-04-14", CHATANYWHERE_CLIENT),
    "gpt-3.5-turbo-0125": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-3.5-turbo-0125", CHATANYWHERE_CLIENT),
    "gpt-4o-2024-11-20": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-4o-2024-11-20", CHATANYWHERE_CLIENT),
    "gpt-4-turbo-2024-04-09": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-4-turbo-2024-04-09", CHATANYWHERE_CLIENT),
    "o3-mini": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "o3-mini", CHATANYWHERE_CLIENT),
    "o4-mini": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "o4-mini", CHATANYWHERE_CLIENT),
    "gemini-1.5-flash-latest": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gemini-1.5-flash-latest", CHATANYWHERE_CLIENT),
    "gemini-1.5-pro-latest": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gemini-1.5-pro-latest", CHATANYWHERE_CLIENT),
    "gemini-2.0-flash-exp": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gemini-2.0-flash-exp", CHATANYWHERE_CLIENT),
    "gpt-5": mk('chatanywhere', CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-5", CHATANYWHERE_CLIENT),
    "gpt-5-nano": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-5-nano", CHATANYWHERE_CLIENT),
    "gpt-5-mini": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gpt-5-mini", CHATANYWHERE_CLIENT),
    'claude-3.5-sonnet-20241022': mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "claude-3-5-sonnet-20241022", CHATANYWHERE_CLIENT),
    'claude-3.5-haiku-20241022': mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "claude-3-5-haiku-20241022", CHATANYWHERE_CLIENT),
    "claude-3.7-sonnet": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "claude-3-7-sonnet-20250219", CHATANYWHERE_CLIENT),
    "claude-sonnet-4": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "claude-sonnet-4-20250514", CHATANYWHERE_CLIENT),
    "deepseek-r1-0528": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL,  "deepseek-reasoner", CHATANYWHERE_CLIENT),
    "deepseek-r1": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "deepseek-r1", CHATANYWHERE_CLIENT),
    "gemini-2.5-pro-05-06": mk("chatanywhere", CHATANYWHERE_KEY, CHATANYWHERE_URL, "gemini-2.5-pro-preview-05-06", CHATANYWHERE_CLIENT),
    # OpenRouter (multi-provider router)
    # "claude-3.7-sonnet": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "anthropic/claude-3.7-sonnet", OPENROUTER_CLIENT),
    # "claude-3.5-sonnet": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "anthropic/claude-3.5-sonnet", OPENROUTER_CLIENT),
    # "claude-3.5-haiku": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "anthropic/claude-3.5-haiku-20241022", OPENROUTER_CLIENT),
    # "gemini-2.5-pro-preview-05-06": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "google/gemini-2.5-pro-preview-05-06", OPENROUTER_CLIENT),
    "llama-4-scout-free": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "meta-llama/llama-4-scout:free", OPENROUTER_CLIENT),
    "deepseek-r1-free": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "deepseek/deepseek-r1:free", OPENROUTER_CLIENT),
    # "gpt-4.1-2025-04-14": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "openai/gpt-4.1-2025-04-14", OPENROUTER_CLIENT),
    "o4-mini-openrouter": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "openai/o4-mini-2025-04-16", OPENROUTER_CLIENT),
    "o3-mini-openrouter": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "openai/o3-mini", OPENROUTER_CLIENT),
    "gpt-4-turbo-openrouter": mk("openrouter", OPENROUTER_KEY, OPENROUTER_URL, "openai/gpt-4-turbo", OPENROUTER_CLIENT),

    # ZhiPu (GLM)
    "glm-4-flash": mk("zhipu", ZHIPU_KEY, "", "glm-4-flash", ZHIPU_CLIENT),

    # Ollama (local)
    "qwen3-32b-local": mk("ollama", "ollama", "http://localhost:11434/v1", "qwen3:32b", OLLAMA_CLIENT),
}