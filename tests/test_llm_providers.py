"""Tests for Ollama provider and Azure OpenAI provider."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
import json

from agentic_chain.llm import (
    LLMFactory,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMUsage,
)
from agentic_chain.llm.base import MessageRole, LLMError
from agentic_chain.llm.ollama_provider import OllamaProvider
from agentic_chain.llm.azure_provider import AzureOpenAIProvider


class TestOllamaProvider:
    """Test cases for Ollama provider."""
    
    def test_init_with_default_host(self):
        """Test initialization with default localhost."""
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        assert provider.api_base == "http://localhost:11434"
    
    @patch.dict('os.environ', {'OLLAMA_HOST': 'http://remote:11434'})
    def test_init_with_env_host(self):
        """Test initialization with environment variable."""
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        assert provider.api_base == "http://remote:11434"
    
    def test_init_with_custom_host(self):
        """Test initialization with custom host."""
        config = LLMConfig(
            provider="ollama",
            model="llama3",
            api_base="http://custom:11434"
        )
        provider = OllamaProvider(config)
        assert provider.api_base == "http://custom:11434"
    
    def test_token_estimation(self):
        """Test token count estimation."""
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        # ~4 characters per token
        text = "Hello world, this is a test."
        tokens = provider._count_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4
    
    @patch('urllib.request.urlopen')
    def test_complete_success(self, mock_urlopen):
        """Test successful completion."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "model": "llama3",
            "message": {"content": "Hello! How can I help?"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        messages = [LLMMessage(role=MessageRole.USER, content="Hi")]
        
        response = provider.complete(messages)
        
        assert response.content == "Hello! How can I help?"
        assert response.model == "llama3"
        assert response.usage.estimated_cost == 0.0  # Local model is free
    
    @patch('urllib.request.urlopen')
    def test_list_models(self, mock_urlopen):
        """Test listing available models."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "models": [
                {"name": "llama3"},
                {"name": "mistral"},
                {"name": "codellama"},
            ]
        }).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        models = provider.list_models()
        
        assert "llama3" in models
        assert "mistral" in models
    
    @patch('urllib.request.urlopen')
    def test_is_available(self, mock_urlopen):
        """Test availability check."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        assert provider.is_available() is True
    
    @patch('urllib.request.urlopen')
    def test_is_not_available(self, mock_urlopen):
        """Test unavailable check."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        assert provider.is_available() is False
    
    def test_handle_connection_error(self):
        """Test handling of connection errors."""
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        error = provider._handle_error(Exception("connection refused"))
        
        assert "Could not connect" in str(error)
    
    def test_handle_model_not_found(self):
        """Test handling of model not found error."""
        provider = OllamaProvider(LLMConfig(provider="ollama", model="llama3"))
        error = provider._handle_error(Exception("model llama3 not found"))
        
        assert "Model not found" in str(error)


class TestAzureOpenAIProvider:
    """Test cases for Azure OpenAI provider."""
    
    def test_init_with_config(self):
        """Test initialization with config values."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4-deployment",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
        )
        provider = AzureOpenAIProvider(config)
        
        assert provider.api_key == "test-key"
        assert provider.api_base == "https://test.openai.azure.com/"
    
    @patch.dict('os.environ', {
        'AZURE_OPENAI_API_KEY': 'env-key',
        'AZURE_OPENAI_ENDPOINT': 'https://env.openai.azure.com/',
    })
    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        provider = AzureOpenAIProvider(LLMConfig(provider="azure", model="gpt-4"))
        
        assert provider.api_key == "env-key"
        assert provider.api_base == "https://env.openai.azure.com/"
    
    def test_init_with_custom_api_version(self):
        """Test initialization with custom API version."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
            extra_options={"api_version": "2024-03-01"},
        )
        provider = AzureOpenAIProvider(config)
        
        assert provider.api_version == "2024-03-01"
    
    def test_default_api_version(self):
        """Test default API version."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
        )
        provider = AzureOpenAIProvider(config)
        
        assert provider.api_version == AzureOpenAIProvider.DEFAULT_API_VERSION
    
    def test_cost_calculation(self):
        """Test cost calculation for Azure models."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
        )
        provider = AzureOpenAIProvider(config)
        
        cost = provider._calculate_cost(1000, 500, "gpt-4")
        # GPT-4: $0.03/1K input, $0.06/1K output
        expected = (1000 / 1000 * 0.03) + (500 / 1000 * 0.06)
        assert abs(cost - expected) < 0.001
    
    def test_handle_rate_limit_error(self):
        """Test handling of rate limit errors."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
        )
        provider = AzureOpenAIProvider(config)
        
        error = provider._handle_error(Exception("rate_limit exceeded"))
        from agentic_chain.llm.base import LLMRateLimitError
        assert isinstance(error, LLMRateLimitError)
    
    def test_handle_auth_error(self):
        """Test handling of authentication errors."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            api_key="test-key",
        )
        provider = AzureOpenAIProvider(config)
        
        error = provider._handle_error(Exception("authentication failed"))
        from agentic_chain.llm.base import LLMAuthenticationError
        assert isinstance(error, LLMAuthenticationError)
    
    def test_handle_deployment_not_found(self):
        """Test handling of deployment not found error."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4-deployment",
            api_key="test-key",
        )
        provider = AzureOpenAIProvider(config)
        
        error = provider._handle_error(Exception("deployment not found"))
        assert "Deployment not found" in str(error)


class TestLLMFactoryWithNewProviders:
    """Test cases for LLMFactory with new providers."""
    
    def test_list_providers_includes_new(self):
        """Test that new providers are listed."""
        providers = LLMFactory.list_providers()
        assert "ollama" in providers
        assert "azure" in providers
        assert "azure_openai" in providers
    
    def test_create_ollama(self):
        """Test creating Ollama provider."""
        provider = LLMFactory.create("ollama", model="llama3")
        assert isinstance(provider, OllamaProvider)
        assert provider.config.model == "llama3"
    
    def test_create_azure(self):
        """Test creating Azure OpenAI provider."""
        provider = LLMFactory.create(
            "azure",
            model="gpt-4-deployment",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
        )
        assert isinstance(provider, AzureOpenAIProvider)
    
    def test_create_azure_openai_alias(self):
        """Test creating Azure provider with alias."""
        provider = LLMFactory.create(
            "azure_openai",
            model="gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
        )
        assert isinstance(provider, AzureOpenAIProvider)
    
    @patch.dict('os.environ', {
        'AZURE_OPENAI_API_KEY': 'azure-key',
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
    }, clear=True)
    def test_create_from_env_azure(self):
        """Test creating provider from environment (Azure)."""
        provider = LLMFactory.create_from_env()
        assert isinstance(provider, AzureOpenAIProvider)
    
    @patch.dict('os.environ', {'OLLAMA_HOST': 'http://localhost:11434'}, clear=True)
    def test_create_from_env_ollama(self):
        """Test creating provider from environment (Ollama)."""
        provider = LLMFactory.create_from_env()
        assert isinstance(provider, OllamaProvider)
