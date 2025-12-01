"""Tests for the base LLM provider classes and utilities."""

import pytest

from agentic_chain.llm.base import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    MessageRole,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMContextLengthError,
)


class TestMessageRole:
    """Test MessageRole enum."""
    
    def test_role_values(self):
        """Test role enum values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestLLMMessage:
    """Test LLMMessage dataclass."""
    
    def test_create_message(self):
        """Test creating a message."""
        msg = LLMMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
    
    def test_to_dict(self):
        """Test converting message to dict."""
        msg = LLMMessage(role=MessageRole.SYSTEM, content="You are helpful")
        result = msg.to_dict()
        
        assert result["role"] == "system"
        assert result["content"] == "You are helpful"


class TestLLMUsage:
    """Test LLMUsage dataclass."""
    
    def test_create_usage(self):
        """Test creating usage object."""
        usage = LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.002,
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.estimated_cost == 0.002
    
    def test_add_usage(self):
        """Test adding usage objects."""
        usage1 = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=0.001)
        usage2 = LLMUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, estimated_cost=0.002)
        
        combined = usage1 + usage2
        
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450
        assert combined.estimated_cost == 0.003
    
    def test_default_values(self):
        """Test default values."""
        usage = LLMUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.estimated_cost == 0.0


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_create_response(self):
        """Test creating a response."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.001)
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            usage=usage,
            finish_reason="stop",
        )
        
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.usage == usage
        assert response.finish_reason == "stop"
    
    def test_to_dict(self):
        """Test converting response to dict."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.001)
        response = LLMResponse(content="Test", model="gpt-4", usage=usage)
        
        result = response.to_dict()
        
        assert result["content"] == "Test"
        assert result["model"] == "gpt-4"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert result["usage"]["estimated_cost"] == 0.001


class TestLLMConfig:
    """Test LLMConfig dataclass."""
    
    def test_create_config(self):
        """Test creating config."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.8,
            max_tokens=2000,
        )
        
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.8
        assert config.max_tokens == 2000
    
    def test_default_model_openai(self):
        """Test default model for OpenAI."""
        config = LLMConfig(provider="openai")
        assert config.model == "gpt-4"
    
    def test_default_model_anthropic(self):
        """Test default model for Anthropic."""
        config = LLMConfig(provider="anthropic")
        assert config.model == "claude-3-sonnet-20240229"
    
    def test_default_model_local(self):
        """Test default model for local."""
        config = LLMConfig(provider="local")
        assert config.model == "llama2"
    
    def test_default_values(self):
        """Test default config values."""
        config = LLMConfig(provider="openai", model="gpt-4")
        
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.extra_options == {}


class TestLLMErrors:
    """Test LLM error classes."""
    
    def test_base_error(self):
        """Test base LLM error."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert isinstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = LLMRateLimitError("Rate limited", retry_after=60.0)
        assert str(error) == "Rate limited"
        assert error.retry_after == 60.0
        assert isinstance(error, LLMError)
    
    def test_rate_limit_error_no_retry(self):
        """Test rate limit error without retry_after."""
        error = LLMRateLimitError("Rate limited")
        assert error.retry_after is None
    
    def test_authentication_error(self):
        """Test authentication error."""
        error = LLMAuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, LLMError)
    
    def test_context_length_error(self):
        """Test context length error."""
        error = LLMContextLengthError("Context too long")
        assert str(error) == "Context too long"
        assert isinstance(error, LLMError)
