"""
LLM Factory - Factory for creating LLM providers.
"""

import os
from typing import Optional

from .base import LLMProvider, LLMConfig, LLMError
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


class LLMFactory:
    """Factory for creating LLM providers."""
    
    # Mapping of provider names to classes
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a custom LLM provider.
        
        Args:
            name: Provider name.
            provider_class: Provider class that inherits from LLMProvider.
        """
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(f"Provider class must inherit from LLMProvider")
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider: Provider name ("openai", "anthropic", etc.).
            model: Model name (optional, uses provider default if not specified).
            api_key: API key (optional, uses environment variable if not specified).
            **kwargs: Additional configuration options.
            
        Returns:
            Configured LLM provider instance.
            
        Raises:
            LLMError: If provider is not supported.
        """
        provider_name = provider.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise LLMError(
                f"Unknown LLM provider: {provider}. "
                f"Available providers: {available}"
            )
        
        # Build configuration
        config = LLMConfig(
            provider=provider_name,
            model=model or "",
            api_key=api_key,
            **kwargs,
        )
        
        # Create and return provider
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_from_config(cls, config: LLMConfig) -> LLMProvider:
        """
        Create an LLM provider from a configuration object.
        
        Args:
            config: LLM configuration.
            
        Returns:
            Configured LLM provider instance.
        """
        provider_name = config.provider.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise LLMError(
                f"Unknown LLM provider: {config.provider}. "
                f"Available providers: {available}"
            )
        
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def create_from_env(cls) -> Optional[LLMProvider]:
        """
        Create an LLM provider based on environment variables.
        
        Checks for API keys in order: OpenAI, Anthropic.
        
        Returns:
            Configured LLM provider, or None if no API keys found.
        """
        # Try OpenAI first
        if os.environ.get("OPENAI_API_KEY"):
            return cls.create("openai")
        
        # Try Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            return cls.create("anthropic")
        
        return None
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List available LLM providers.
        
        Returns:
            List of provider names.
        """
        return list(cls._providers.keys())
