"""
Ollama Provider - Implementation for local Ollama API.

This provider enables running LLM inference locally using Ollama,
providing privacy-preserving AI capabilities without API costs.
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


class OllamaProvider(LLMProvider):
    """
    Ollama local API provider implementation.
    
    Ollama allows running large language models locally without
    requiring API keys or internet connectivity.
    
    Features:
    - Zero API cost
    - Complete data privacy
    - No rate limiting
    - Offline capability
    
    Supported models include: llama2, llama3, mistral, codellama, etc.
    
    Example:
        >>> from agentic_chain.llm import LLMFactory
        >>> provider = LLMFactory.create("ollama", model="llama3")
        >>> response = provider.complete(messages)
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the Ollama provider.
        
        Args:
            config: Configuration for the provider.
        """
        super().__init__(config)
        
        # Default to localhost if no API base provided
        self.api_base = config.api_base or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        self._client = None
    
    def _get_client(self):
        """Get or create the HTTP client for Ollama API."""
        if self._client is None:
            try:
                import urllib.request
                # We'll use urllib for simple HTTP requests to avoid extra dependencies
                self._client = True  # Mark client as initialized
            except ImportError:
                raise LLMError("HTTP client not available")
        return self._client
    
    def _make_request(
        self,
        endpoint: str,
        data: dict,
        stream: bool = False,
    ) -> dict:
        """Make an HTTP request to the Ollama API."""
        import urllib.request
        import urllib.error
        
        url = f"{self.api_base}{endpoint}"
        json_data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(
                req,
                timeout=self.config.timeout,
            ) as response:
                if stream:
                    # For streaming, return the response object
                    return response
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            raise self._handle_error(Exception(f"HTTP {e.code}: {error_body}"))
        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not connect to Ollama at {self.api_base}. "
                f"Make sure Ollama is running: {e}"
            )
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Note: This is a rough approximation (~4 characters per token for English).
        Actual token counts vary significantly by model and tokenizer.
        For accurate token counting, consider using the tiktoken library
        or the specific model's tokenizer.
        
        Ollama provides actual token counts in the response when available,
        which are used preferentially.
        """
        # Simple approximation: ~4 characters per token for English
        # This can be off by 20-50% for non-English text or code
        return max(1, len(text) // 4)
    
    def complete(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion using Ollama.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional Ollama-specific options.
            
        Returns:
            LLMResponse with the generated content.
        """
        self._get_client()
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        # Build request parameters
        model = kwargs.get("model", self.config.model)
        request_params = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }
        
        # Add max tokens if specified
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens:
            request_params["options"]["num_predict"] = max_tokens
        
        # Add extra options
        for key, value in self.config.extra_options.items():
            if key not in request_params:
                request_params["options"][key] = value
        
        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._make_request("/api/chat", request_params)
                
                # Extract content
                content = response.get("message", {}).get("content", "")
                
                # Calculate token usage (Ollama provides these in response)
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                
                # If not provided, estimate
                if prompt_tokens == 0:
                    prompt_text = " ".join(m["content"] for m in ollama_messages)
                    prompt_tokens = self._count_tokens(prompt_text)
                if completion_tokens == 0:
                    completion_tokens = self._count_tokens(content)
                
                total_tokens = prompt_tokens + completion_tokens
                
                # Ollama is free, so cost is always 0
                usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost=0.0,  # Local models are free
                )
                
                # Update total usage
                self._total_usage = self._total_usage + usage
                
                return LLMResponse(
                    content=content,
                    model=response.get("model", model),
                    usage=usage,
                    finish_reason=response.get("done_reason", "stop"),
                    raw_response=response,
                )
                
            except LLMError:
                raise
            except Exception as e:
                last_error = self._handle_error(e)
                if attempt < self.config.max_retries - 1:
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
        Stream a completion using Ollama.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional Ollama-specific options.
            
        Yields:
            Content chunks as they are generated.
        """
        import urllib.request
        
        self._get_client()
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        # Build request parameters
        model = kwargs.get("model", self.config.model)
        request_params = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }
        
        url = f"{self.api_base}/api/chat"
        json_data = json.dumps(request_params).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(
                req,
                timeout=self.config.timeout,
            ) as response:
                for line in response:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise self._handle_error(e)
    
    def list_models(self) -> list[str]:
        """
        List available models on the local Ollama instance.
        
        Returns:
            List of available model names.
        """
        import urllib.request
        
        url = f"{self.api_base}/api/tags"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and running.
        
        Returns:
            True if Ollama is accessible, False otherwise.
        """
        import urllib.request
        
        url = f"{self.api_base}/api/tags"
        
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama library.
        
        Args:
            model_name: Name of the model to pull.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            self._make_request("/api/pull", {"name": model_name})
            return True
        except Exception as e:
            logger.warning(f"Could not pull model {model_name}: {e}")
            return False
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Convert Ollama errors to LLM errors."""
        error_message = str(error)
        
        # Check for specific error types
        if "connection refused" in error_message.lower():
            return LLMError(
                f"Could not connect to Ollama. Is it running? ({self.api_base})"
            )
        
        if "model" in error_message.lower() and "not found" in error_message.lower():
            return LLMError(
                f"Model not found. Run 'ollama pull {self.config.model}' to download it."
            )
        
        if "context" in error_message.lower():
            return LLMContextLengthError(error_message)
        
        return LLMError(error_message)
