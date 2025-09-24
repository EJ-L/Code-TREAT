# ==============================
# File: models/providers.py
# Central provider registry + factory
# ==============================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .apibase import APIBase


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    client_library: str
    supported_models: List[str] = field(default_factory=list)


class ProviderRegistry:
    _registry: Dict[str, ProviderSpec] = {}

    @classmethod
    def register(cls, spec: ProviderSpec) -> None:
        key = spec.name.lower()
        if key in cls._registry:
            # Overwrite allowed so you can hot-reload during research
            pass
        cls._registry[key] = spec

    @classmethod
    def get(cls, name: str) -> ProviderSpec:
        spec = cls._registry.get(name.lower())
        if not spec:
            raise ValueError(f"Unknown provider: {name}")
        return spec


class ProviderFactory:
    @staticmethod
    def create(provider_name: str, api_key: str, base_url: str, model_name: str, client: Optional[str] = None) -> APIBase:
        """
        Create an APIBase instance configured for a provider.
        - provider_name: e.g. "openai", "deepinfra", "anthropic", "openrouter", "chatanywhere", "ollama", "alibaba".
        - client: if provided, overrides the registry's client_library (keeps backward compatibility with your signature).
        """
        spec = ProviderRegistry.get(provider_name)
        # Validate model (if the provider lists any)
        if spec.supported_models and (model_name not in spec.supported_models):
            raise ValueError(
                f"Invalid model '{model_name}' for provider '{provider_name}'. "
                f"Choose from: {spec.supported_models}"
            )
        
        api = APIBase(api_key_env=api_key, base_url=base_url, api_model_name=model_name)
        api.setup_client(client or spec.client_library)
        return api


# ---- Default registrations (edit here, not in N files) ----
# Keep this list short at first; you can expand as you standardize.
ProviderRegistry.register(ProviderSpec(
    name="openai",
    client_library="OPENAI_CLIENT",
    supported_models=[
        # Put explicit allowlist here ONLY if you want validation. Otherwise, leave empty to accept any.
        # Example placeholder:
        # "openai/gpt-4o-2024-11-20", "gpt-4.1-2025-04-14", "o4-mini", "o3-mini"
    ],
))

ProviderRegistry.register(ProviderSpec(
    name="openrouter",
    client_library="OPENROUTER_CLIENT",
    supported_models=[],  # accept any, or populate
))

ProviderRegistry.register(ProviderSpec(
    name="deepinfra",
    client_library="DEEPINFRA_CLIENT",
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="deepseek",
    client_library="DEEPSEEK_CLIENT",
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="anthropic",
    client_library="ANTHROPIC_CLIENT",
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="google",
    client_library="GOOGLE_CLIENT",
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="ollama",
    client_library="OLLAMA",
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="chatanywhere",
    client_library="CHATANYWHERE_CLIENT",  # it uses OpenAI-compatible endpoint in your base
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="alibaba",
    client_library="OPENAI_CLIENT",  # set to whatever client you actually use
    supported_models=[],
))

ProviderRegistry.register(ProviderSpec(
    name="zhipu",
    client_library="ZHIPU_CLIENT",
    supported_models=[],
))