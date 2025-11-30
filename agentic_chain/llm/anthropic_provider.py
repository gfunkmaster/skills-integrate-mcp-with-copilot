"""
Anthropic Provider - Implementation for Anthropic Claude API.
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
    MessageRole,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMContextLengthError,
)


logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Configuration for the provider.
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.api_base = config.api_base or "https://api.anthropic.com"
        self._client = None
    
    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise LLMError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client
    
    def _prepare_messages(
        self,
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict]]:
        """
        Prepare messages for Anthropic API.
        
        Anthropic uses a separate system parameter, so we need to extract
        the system message from the conversation.
        
        Args:
            messages: List of messages.
            
        Returns:
            Tuple of (system_message, conversation_messages).
        """
        system_message = ""
        conversation = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message += msg.content + "\n"
            else:
                # Map roles to Anthropic format
                role = "user" if msg.role == MessageRole.USER else "assistant"
                conversation.append({
                    "role": role,
                    "content": msg.content,
                })
        
        return system_message.strip(), conversation
    
    def complete(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion using Anthropic Claude.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional Anthropic-specific options.
            
        Returns:
            LLMResponse with the generated content.
        """
        client = self._get_client()
        
        # Prepare messages
        system_message, conversation = self._prepare_messages(messages)
        
        # Build request parameters
        request_params = {
            "model": kwargs.get("model", self.config.model),
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Add system message if present
        if system_message:
            request_params["system"] = system_message
        
        # Add temperature if not default
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature != 1.0:  # Anthropic default is 1.0
            request_params["temperature"] = temperature
        
        # Add extra options
        for key, value in self.config.extra_options.items():
            if key not in request_params:
                request_params[key] = value
        
        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = client.messages.create(**request_params)
                
                # Extract usage
                usage_data = response.usage
                prompt_tokens = usage_data.input_tokens if usage_data else 0
                completion_tokens = usage_data.output_tokens if usage_data else 0
                total_tokens = prompt_tokens + completion_tokens
                
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
                
                # Extract content
                content = ""
                if response.content:
                    for block in response.content:
                        if hasattr(block, "text"):
                            content += block.text
                
                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=response.stop_reason,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
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
        Stream a completion using Anthropic Claude.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional Anthropic-specific options.
            
        Yields:
            Content chunks as they are generated.
        """
        client = self._get_client()
        
        # Prepare messages
        system_message, conversation = self._prepare_messages(messages)
        
        # Build request parameters
        request_params = {
            "model": kwargs.get("model", self.config.model),
            "messages": conversation,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        if system_message:
            request_params["system"] = system_message
        
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature != 1.0:
            request_params["temperature"] = temperature
        
        try:
            with client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Convert Anthropic errors to LLM errors."""
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
            if hasattr(error, "response") and hasattr(error.response, "headers"):
                retry_after_str = error.response.headers.get("Retry-After")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            return LLMRateLimitError(error_message, retry_after)
        
        if "authentication" in error_message.lower() or "api_key" in error_message.lower():
            return LLMAuthenticationError(error_message)
        
        if "context" in error_message.lower() and "length" in error_message.lower():
            return LLMContextLengthError(error_message)
        
        return LLMError(error_message)
