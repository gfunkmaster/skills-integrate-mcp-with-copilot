"""
OpenAI Provider - Implementation for OpenAI API.
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


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Configuration for the provider.
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        self.api_base = config.api_base or "https://api.openai.com/v1"
        self._client = None
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise LLMError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    def complete(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional OpenAI-specific options.
            
        Returns:
            LLMResponse with the generated content.
        """
        client = self._get_client()
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        # Merge config with kwargs
        request_params = {
            "model": kwargs.get("model", self.config.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Add extra options
        for key, value in self.config.extra_options.items():
            if key not in request_params:
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
                
                # Calculate cost
                cost = self._calculate_cost(
                    prompt_tokens,
                    completion_tokens,
                    request_params["model"],
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
                    wait_time = last_error.retry_after or (self.config.retry_delay * (2 ** attempt))
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
        Stream a completion using OpenAI.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional OpenAI-specific options.
            
        Yields:
            Content chunks as they are generated.
        """
        client = self._get_client()
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        # Merge config with kwargs
        request_params = {
            "model": kwargs.get("model", self.config.model),
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
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Convert OpenAI errors to LLM errors."""
        error_message = str(error)
        
        # Try to parse error details
        if hasattr(error, "response"):
            try:
                response_json = error.response.json()
                error_message = response_json.get("error", {}).get("message", error_message)
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Check for specific error types
        if "rate_limit" in error_message.lower():
            retry_after = None
            if hasattr(error, "response") and error.response.headers:
                retry_after_str = error.response.headers.get("Retry-After")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            return LLMRateLimitError(error_message, retry_after)
        
        if "authentication" in error_message.lower() or "api_key" in error_message.lower():
            return LLMAuthenticationError(error_message)
        
        if "context_length" in error_message.lower() or "maximum context" in error_message.lower():
            return LLMContextLengthError(error_message)
        
        return LLMError(error_message)
