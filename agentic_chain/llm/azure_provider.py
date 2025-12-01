"""
Azure OpenAI Provider - Implementation for Azure OpenAI Service.

This provider enables using Azure-hosted OpenAI models with enterprise
security, compliance, and regional availability features.
"""

import json
import logging
import os
import time
from typing import Any, Iterator, Optional

from .base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMUsage,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMContextLengthError,
)


logger = logging.getLogger(__name__)


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI Service provider implementation.
    
    Azure OpenAI provides enterprise-grade security, compliance,
    and regional data residency for OpenAI models.
    
    Features:
    - Enterprise security and compliance
    - Regional data residency
    - Virtual network support
    - Managed identity authentication
    
    Required configuration:
    - api_base: Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
    - api_key: Azure OpenAI API key
    - model: Deployment name (not the model name)
    
    Optional configuration via extra_options:
    - api_version: API version (default: 2024-02-01)
    
    Example:
        >>> from agentic_chain.llm import LLMFactory
        >>> provider = LLMFactory.create(
        ...     "azure",
        ...     api_base="https://your-resource.openai.azure.com/",
        ...     model="gpt-4-deployment",
        ...     api_key="your-api-key",
        ... )
        >>> response = provider.complete(messages)
    """
    
    DEFAULT_API_VERSION = "2024-02-01"
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the Azure OpenAI provider.
        
        Args:
            config: Configuration for the provider.
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "No Azure OpenAI API key provided. "
                "Set AZURE_OPENAI_API_KEY environment variable."
            )
        
        # Get endpoint from config or environment
        self.api_base = config.api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.api_base:
            logger.warning(
                "No Azure OpenAI endpoint provided. "
                "Set AZURE_OPENAI_ENDPOINT environment variable."
            )
        
        # API version
        self.api_version = config.extra_options.get(
            "api_version",
            os.environ.get("AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION),
        )
        
        self._client = None
    
    def _get_client(self):
        """Get or create the Azure OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.api_base,
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise LLMError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
            except Exception as e:
                raise LLMError(f"Failed to create Azure OpenAI client: {e}")
        return self._client
    
    def complete(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion using Azure OpenAI.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional Azure OpenAI-specific options.
            
        Returns:
            LLMResponse with the generated content.
        """
        client = self._get_client()
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        # In Azure OpenAI, the model parameter is the deployment name
        deployment_name = kwargs.get("model", self.config.model)
        
        # Merge config with kwargs
        request_params = {
            "model": deployment_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Add extra options
        for key, value in self.config.extra_options.items():
            if key not in request_params and key != "api_version":
                request_params[key] = value
        
        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(**request_params)
                
                # Extract usage
                usage_data = response.usage
                prompt_tokens = usage_data.prompt_tokens if usage_data else 0
                completion_tokens = usage_data.completion_tokens if usage_data else 0
                total_tokens = usage_data.total_tokens if usage_data else 0
                
                # Calculate cost (Azure pricing may vary by region)
                cost = self._calculate_cost(
                    prompt_tokens,
                    completion_tokens,
                    deployment_name,
                )
                
                usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost=cost,
                )
                
                # Update total usage
                self._total_usage = self._total_usage + usage
                
                # Build response
                choice = response.choices[0] if response.choices else None
                content = choice.message.content if choice and choice.message else ""
                finish_reason = choice.finish_reason if choice else None
                
                # Serialize raw response if supported (Pydantic v2+)
                raw_response = None
                try:
                    raw_response = response.model_dump()
                except AttributeError:
                    # Fallback for older versions
                    pass
                
                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    raw_response=raw_response,
                )
                
            except Exception as e:
                last_error = self._handle_error(e)
                if isinstance(last_error, LLMRateLimitError):
                    wait_time = last_error.retry_after or (
                        self.config.retry_delay * (2 ** attempt)
                    )
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise last_error
        
        raise last_error or LLMError("Max retries exceeded")
    
    def stream(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream a completion using Azure OpenAI.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional Azure OpenAI-specific options.
            
        Yields:
            Content chunks as they are generated.
        """
        client = self._get_client()
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        # In Azure OpenAI, the model parameter is the deployment name
        deployment_name = kwargs.get("model", self.config.model)
        
        # Merge config with kwargs
        request_params = {
            "model": deployment_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        
        try:
            stream = client.chat.completions.create(**request_params)
            
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        
        except Exception as e:
            raise self._handle_error(e)
    
    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """
        Calculate estimated cost for the request.
        
        WARNING: Azure pricing varies significantly by:
        - Region (East US vs. Europe)
        - Agreement type (Pay-as-you-go vs. Enterprise)
        - Reserved capacity commitments
        
        The default prices here are approximate pay-as-you-go US prices
        as of 2024. For accurate cost tracking:
        - Set custom pricing via config.extra_options['pricing']
        - Or use Azure Cost Management for authoritative data
        
        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            model: Deployment/model name.
            
        Returns:
            Estimated cost in USD (may not reflect actual billing).
        """
        # Check for custom pricing in config
        custom_pricing = self.config.extra_options.get("pricing", {})
        
        # Default Azure OpenAI pricing per 1000 tokens (approximate, US pay-as-you-go)
        # WARNING: These prices may be outdated. Check Azure pricing page for current rates.
        # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
        default_pricing = {
            "gpt-4": (0.03, 0.06),
            "gpt-4-32k": (0.06, 0.12),
            "gpt-35-turbo": (0.0015, 0.002),
            "gpt-35-turbo-16k": (0.003, 0.004),
        }
        
        # Merge with custom pricing (custom pricing takes precedence)
        pricing = {**default_pricing, **custom_pricing}
        
        # Find matching pricing
        model_lower = model.lower()
        for model_prefix, (input_price, output_price) in pricing.items():
            if model_prefix in model_lower:
                input_cost = (prompt_tokens / 1000) * input_price
                output_cost = (completion_tokens / 1000) * output_price
                return input_cost + output_cost
        
        # Return 0 if model not found (no estimate available)
        logger.debug(
            f"No pricing data for model '{model}'. "
            "Set custom pricing via config.extra_options['pricing']."
        )
        return 0.0
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Convert Azure OpenAI errors to LLM errors."""
        error_message = str(error)
        
        # Try to parse error details
        if hasattr(error, "response"):
            try:
                response_json = error.response.json()
                error_message = response_json.get("error", {}).get(
                    "message", error_message
                )
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Check for specific error types
        if "rate_limit" in error_message.lower() or "429" in error_message:
            retry_after = None
            if hasattr(error, "response") and error.response.headers:
                retry_after_str = error.response.headers.get("Retry-After")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            return LLMRateLimitError(error_message, retry_after)
        
        if (
            "authentication" in error_message.lower()
            or "api_key" in error_message.lower()
            or "401" in error_message
        ):
            return LLMAuthenticationError(error_message)
        
        if (
            "context_length" in error_message.lower()
            or "maximum context" in error_message.lower()
        ):
            return LLMContextLengthError(error_message)
        
        if "deployment" in error_message.lower() and "not found" in error_message.lower():
            return LLMError(
                f"Deployment not found: {self.config.model}. "
                "Make sure the deployment name matches your Azure configuration."
            )
        
        return LLMError(error_message)
