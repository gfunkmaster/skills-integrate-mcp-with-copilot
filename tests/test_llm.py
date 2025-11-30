"""Tests for the LLM integration module."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from agentic_chain.llm import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMUsage,
    LLMFactory,
    OpenAIProvider,
    AnthropicProvider,
)
from agentic_chain.llm.base import (
    MessageRole,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
)


class TestLLMConfig:
    """Test cases for LLMConfig."""
    
    def test_default_model_openai(self):
        """Test default model for OpenAI provider."""
        config = LLMConfig(provider="openai")
        assert config.model == "gpt-4"
    
    def test_default_model_anthropic(self):
        """Test default model for Anthropic provider."""
        config = LLMConfig(provider="anthropic")
        assert config.model == "claude-3-sonnet-20240229"
    
    def test_custom_model(self):
        """Test custom model specification."""
        config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
        assert config.model == "gpt-3.5-turbo"
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig(provider="openai")
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.max_retries == 3


class TestLLMMessage:
    """Test cases for LLMMessage."""
    
    def test_to_dict(self):
        """Test message serialization."""
        msg = LLMMessage(role=MessageRole.USER, content="Hello")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello"}
    
    def test_system_role(self):
        """Test system role message."""
        msg = LLMMessage(role=MessageRole.SYSTEM, content="You are helpful")
        assert msg.role == MessageRole.SYSTEM


class TestLLMUsage:
    """Test cases for LLMUsage."""
    
    def test_addition(self):
        """Test adding usage stats."""
        usage1 = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=0.01)
        usage2 = LLMUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, estimated_cost=0.02)
        
        combined = usage1 + usage2
        
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450
        assert combined.estimated_cost == 0.03


class TestLLMResponse:
    """Test cases for LLMResponse."""
    
    def test_to_dict(self):
        """Test response serialization."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.001)
        response = LLMResponse(
            content="Hello world",
            model="gpt-4",
            usage=usage,
            finish_reason="stop"
        )
        
        result = response.to_dict()
        
        assert result["content"] == "Hello world"
        assert result["model"] == "gpt-4"
        assert result["usage"]["total_tokens"] == 15
        assert result["finish_reason"] == "stop"


class TestLLMFactory:
    """Test cases for LLMFactory."""
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = LLMFactory.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
    
    def test_create_openai(self):
        """Test creating OpenAI provider."""
        provider = LLMFactory.create("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.provider == "openai"
    
    def test_create_anthropic(self):
        """Test creating Anthropic provider."""
        provider = LLMFactory.create("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.provider == "anthropic"
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            LLMFactory.create("unknown")
    
    def test_create_with_model(self):
        """Test creating provider with specific model."""
        provider = LLMFactory.create("openai", model="gpt-3.5-turbo", api_key="test-key")
        assert provider.config.model == "gpt-3.5-turbo"
    
    def test_create_from_config(self):
        """Test creating provider from config object."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test-key")
        provider = LLMFactory.create_from_config(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.model == "gpt-4"


class MockOpenAIProvider(LLMProvider):
    """Mock LLM provider for testing without API calls."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._response_content = "Mock response"
    
    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Return a mock response."""
        usage = LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.01,
        )
        self._total_usage = self._total_usage + usage
        
        return LLMResponse(
            content=self._response_content,
            model=self.config.model,
            usage=usage,
            finish_reason="stop",
        )
    
    def stream(self, messages: list[LLMMessage], **kwargs):
        """Return mock streaming response."""
        for word in self._response_content.split():
            yield word + " "


class TestLLMProviderBase:
    """Test cases for LLMProvider base class methods."""
    
    def test_generate_code(self):
        """Test code generation method."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        provider = MockOpenAIProvider(config)
        provider._response_content = "def hello(): print('Hello')"
        
        response = provider.generate_code(
            prompt="Create a hello function",
            language="Python"
        )
        
        assert "def hello" in response.content
        assert response.model == "gpt-4"
    
    def test_review_code(self):
        """Test code review method."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        provider = MockOpenAIProvider(config)
        provider._response_content = "Code looks good, no issues found."
        
        response = provider.review_code(
            code="def hello(): print('Hello')",
            context="A simple greeting function"
        )
        
        assert response.content == "Code looks good, no issues found."
    
    def test_generate_implementation_plan(self):
        """Test implementation plan generation."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        provider = MockOpenAIProvider(config)
        provider._response_content = "1. Analyze the issue\n2. Implement solution"
        
        response = provider.generate_implementation_plan(
            issue_description="Fix login bug",
            project_context={"languages": {"Python": 10}}
        )
        
        assert "Analyze" in response.content
    
    def test_usage_tracking(self):
        """Test that usage is tracked across requests."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        provider = MockOpenAIProvider(config)
        
        # Make multiple requests
        provider.generate_code(prompt="Test 1")
        provider.generate_code(prompt="Test 2")
        
        usage = provider.total_usage
        assert usage.total_tokens == 300  # 150 * 2
        assert usage.prompt_tokens == 200  # 100 * 2
    
    def test_reset_usage(self):
        """Test resetting usage tracking."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        provider = MockOpenAIProvider(config)
        
        provider.generate_code(prompt="Test")
        assert provider.total_usage.total_tokens == 150
        
        provider.reset_usage()
        assert provider.total_usage.total_tokens == 0
    
    def test_cost_calculation(self):
        """Test cost calculation for different models."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")
        provider = MockOpenAIProvider(config)
        
        cost = provider._calculate_cost(1000, 500, "gpt-4")
        # GPT-4: $0.03/1K input, $0.06/1K output
        expected = (1000/1000 * 0.03) + (500/1000 * 0.06)
        assert abs(cost - expected) < 0.001


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = OpenAIProvider(LLMConfig(provider="openai", api_key="test-key"))
        assert provider.api_key == "test-key"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    def test_init_with_env_key(self):
        """Test initialization with environment variable."""
        provider = OpenAIProvider(LLMConfig(provider="openai"))
        assert provider.api_key == "env-key"
    
    def test_init_without_key(self):
        """Test initialization without API key logs warning."""
        with patch.dict('os.environ', {}, clear=True):
            provider = OpenAIProvider(LLMConfig(provider="openai"))
            assert provider.api_key is None


class TestAnthropicProvider:
    """Test cases for Anthropic provider."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = AnthropicProvider(LLMConfig(provider="anthropic", api_key="test-key"))
        assert provider.api_key == "test-key"
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key'})
    def test_init_with_env_key(self):
        """Test initialization with environment variable."""
        provider = AnthropicProvider(LLMConfig(provider="anthropic"))
        assert provider.api_key == "env-key"
    
    def test_prepare_messages(self):
        """Test message preparation for Anthropic format."""
        provider = AnthropicProvider(LLMConfig(provider="anthropic", api_key="test"))
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            LLMMessage(role=MessageRole.USER, content="Hello"),
            LLMMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        
        system, conversation = provider._prepare_messages(messages)
        
        assert system == "You are helpful"
        assert len(conversation) == 2
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"


class TestLLMErrors:
    """Test cases for LLM error handling."""
    
    def test_rate_limit_error(self):
        """Test rate limit error with retry after."""
        error = LLMRateLimitError("Rate limited", retry_after=60.0)
        assert error.retry_after == 60.0
        assert "Rate limited" in str(error)
    
    def test_authentication_error(self):
        """Test authentication error."""
        error = LLMAuthenticationError("Invalid API key")
        assert "Invalid API key" in str(error)
    
    def test_base_error(self):
        """Test base LLM error."""
        error = LLMError("Something went wrong")
        assert "Something went wrong" in str(error)
