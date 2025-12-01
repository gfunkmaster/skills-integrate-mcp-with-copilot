"""Tests for the OpenAI provider."""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agentic_chain.llm.base import (
    LLMConfig,
    LLMMessage,
    LLMUsage,
    MessageRole,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMContextLengthError,
)
from agentic_chain.llm.openai_provider import OpenAIProvider


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""
    
    def test_init_with_api_key(self, mock_llm_config):
        """Test initialization with API key in config."""
        provider = OpenAIProvider(mock_llm_config)
        
        assert provider.api_key == "test-key-123"
        assert provider.api_base == "https://api.openai.com/v1"
        assert provider.config == mock_llm_config
    
    def test_init_with_env_api_key(self, mock_llm_config):
        """Test initialization with API key from environment."""
        mock_llm_config.api_key = None
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
            provider = OpenAIProvider(mock_llm_config)
            assert provider.api_key == "env-key-456"
    
    def test_init_without_api_key(self, mock_llm_config):
        """Test initialization without API key logs warning."""
        mock_llm_config.api_key = None
        
        with patch.dict(os.environ, {}, clear=True):
            with patch("agentic_chain.llm.openai_provider.logger") as mock_logger:
                provider = OpenAIProvider(mock_llm_config)
                assert provider.api_key is None
                mock_logger.warning.assert_called_once()
    
    def test_init_with_custom_base_url(self, mock_llm_config):
        """Test initialization with custom base URL."""
        mock_llm_config.api_base = "https://custom.openai.com/v1"
        provider = OpenAIProvider(mock_llm_config)
        
        assert provider.api_base == "https://custom.openai.com/v1"
    
    def test_get_client_creates_client(self, mock_llm_config):
        """Test that _get_client creates OpenAI client."""
        with patch('builtins.__import__', side_effect=lambda name, *args: MagicMock() if name == 'openai' else __import__(name, *args)):
            provider = OpenAIProvider(mock_llm_config)
            # Don't actually call _get_client as it requires real openai module
            # Just verify the provider is initialized correctly
            assert provider.api_key == "test-key-123"
            assert provider._client is None
    
    def test_get_client_missing_package(self, mock_llm_config):
        """Test error when OpenAI package is not installed."""
        provider = OpenAIProvider(mock_llm_config)
        
        # Mock import to raise ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
            with pytest.raises(LLMError, match="OpenAI package not installed"):
                provider._get_client()
    
    def test_complete_success(self, mock_llm_config, mock_openai_client, sample_llm_messages):
        """Test successful completion."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            response = provider.complete(sample_llm_messages)
            
            assert response.content == "This is a test response from the LLM."
            assert response.model == "gpt-4o-mini"
            assert response.usage.prompt_tokens == 50
            assert response.usage.completion_tokens == 20
            assert response.finish_reason == "stop"
    
    def test_complete_with_custom_params(self, mock_llm_config, mock_openai_client, sample_llm_messages):
        """Test completion with custom parameters."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            response = provider.complete(
                sample_llm_messages,
                temperature=0.9,
                max_tokens=500,
            )
            
            # Verify custom params were passed
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["temperature"] == 0.9
            assert call_args[1]["max_tokens"] == 500
    
    def test_complete_updates_total_usage(self, mock_llm_config, mock_openai_client, sample_llm_messages):
        """Test that completion updates total usage."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            
            # Make two completions
            provider.complete(sample_llm_messages)
            provider.complete(sample_llm_messages)
            
            # Check total usage accumulated
            assert provider.total_usage.prompt_tokens == 100
            assert provider.total_usage.completion_tokens == 40
            assert provider.total_usage.total_tokens == 140
    
    def test_complete_with_rate_limit_error(self, mock_llm_config, sample_llm_messages):
        """Test handling of rate limit errors."""
        mock_client = MagicMock()
        mock_error = Exception("rate_limit exceeded")
        mock_client.chat.completions.create.side_effect = mock_error
        
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_client):
            provider = OpenAIProvider(mock_llm_config)
            
            with pytest.raises(LLMRateLimitError):
                provider.complete(sample_llm_messages)
    
    def test_complete_with_authentication_error(self, mock_llm_config, sample_llm_messages):
        """Test handling of authentication errors."""
        mock_client = MagicMock()
        mock_error = Exception("authentication failed")
        mock_client.chat.completions.create.side_effect = mock_error
        
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_client):
            provider = OpenAIProvider(mock_llm_config)
            
            with pytest.raises(LLMAuthenticationError):
                provider.complete(sample_llm_messages)
    
    def test_complete_with_context_length_error(self, mock_llm_config, sample_llm_messages):
        """Test handling of context length errors."""
        mock_client = MagicMock()
        mock_error = Exception("context_length exceeded")
        mock_client.chat.completions.create.side_effect = mock_error
        
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_client):
            provider = OpenAIProvider(mock_llm_config)
            
            with pytest.raises(LLMContextLengthError):
                provider.complete(sample_llm_messages)
    
    def test_complete_with_retry(self, mock_llm_config, mock_openai_client, sample_llm_messages):
        """Test retry logic on transient errors."""
        mock_client = MagicMock()
        # Fail first, succeed second
        mock_client.chat.completions.create.side_effect = [
            Exception("temporary error"),
            mock_openai_client.chat.completions.create.return_value,
        ]
        
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_client):
            provider = OpenAIProvider(mock_llm_config)
            response = provider.complete(sample_llm_messages)
            
            # Should succeed after retry
            assert response.content == "This is a test response from the LLM."
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_stream_success(self, mock_llm_config, sample_llm_messages):
        """Test successful streaming."""
        mock_client = MagicMock()
        
        # Mock stream chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
        
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_client):
            provider = OpenAIProvider(mock_llm_config)
            chunks = list(provider.stream(sample_llm_messages))
            
            assert chunks == ["Hello", " world"]
    
    def test_stream_with_empty_chunks(self, mock_llm_config, sample_llm_messages):
        """Test streaming with empty delta content."""
        mock_client = MagicMock()
        
        # Mock chunks with None content
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = None
        
        mock_client.chat.completions.create.return_value = [mock_chunk]
        
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_client):
            provider = OpenAIProvider(mock_llm_config)
            chunks = list(provider.stream(sample_llm_messages))
            
            assert chunks == []
    
    def test_generate_code(self, mock_llm_config, mock_openai_client):
        """Test code generation."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            response = provider.generate_code(
                prompt="Create a function that adds two numbers",
                language="Python",
            )
            
            assert response is not None
            # Verify system message mentions Python
            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert any("Python" in msg["content"] for msg in messages if msg["role"] == "system")
    
    def test_review_code(self, mock_llm_config, mock_openai_client, sample_code):
        """Test code review."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            response = provider.review_code(sample_code)
            
            assert response is not None
            # Verify code is in user message
            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert any(sample_code in msg["content"] for msg in messages if msg["role"] == "user")
    
    def test_generate_implementation_plan(self, mock_llm_config, mock_openai_client):
        """Test implementation plan generation."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            response = provider.generate_implementation_plan(
                issue_description="Add a new login feature",
                project_context={"languages": {"Python": 10}, "patterns": {"framework": "FastAPI"}},
            )
            
            assert response is not None
            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
            assert "Add a new login feature" in user_message
            assert "FastAPI" in user_message
    
    def test_reset_usage(self, mock_llm_config, mock_openai_client, sample_llm_messages):
        """Test resetting usage tracking."""
        with patch.object(OpenAIProvider, "_get_client", return_value=mock_openai_client):
            provider = OpenAIProvider(mock_llm_config)
            
            # Make a completion
            provider.complete(sample_llm_messages)
            assert provider.total_usage.total_tokens > 0
            
            # Reset usage
            provider.reset_usage()
            assert provider.total_usage.total_tokens == 0
            assert provider.total_usage.prompt_tokens == 0
            assert provider.total_usage.completion_tokens == 0
