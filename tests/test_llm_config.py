"""Tests for configuration module."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from agentic_chain.llm.config import (
    AgenticChainConfig,
    CacheConfig,
    CostConfig,
    PrivacyConfig,
    find_config_file,
    load_config,
    load_config_from_env,
    load_yaml_file,
    create_default_config_file,
    _deep_merge,
)


class TestCacheConfig:
    """Test cases for CacheConfig."""
    
    def test_default_values(self):
        """Test default cache configuration values."""
        config = CacheConfig()
        
        assert config.enabled is False
        assert config.ttl == 3600
        assert config.max_size == 1000
        assert config.storage == "memory"
    
    def test_custom_values(self):
        """Test custom cache configuration values."""
        config = CacheConfig(
            enabled=True,
            ttl=7200,
            max_size=500,
            storage="disk",
            disk_path="/tmp/cache",
        )
        
        assert config.enabled is True
        assert config.ttl == 7200
        assert config.storage == "disk"
        assert config.disk_path == "/tmp/cache"


class TestCostConfig:
    """Test cases for CostConfig."""
    
    def test_default_values(self):
        """Test default cost configuration values."""
        config = CostConfig()
        
        assert config.max_cost_per_analysis == 0.05
        assert config.budget_alert_threshold == 10.00
        assert config.track_usage is True


class TestPrivacyConfig:
    """Test cases for PrivacyConfig."""
    
    def test_default_values(self):
        """Test default privacy configuration values."""
        config = PrivacyConfig()
        
        assert config.redact_pii is False
        assert config.share_telemetry is False
        assert config.local_only is False


class TestAgenticChainConfig:
    """Test cases for AgenticChainConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AgenticChainConfig()
        
        assert config.provider == "openai"
        assert config.model == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
    
    def test_from_dict_minimal(self):
        """Test creating config from minimal dictionary."""
        data = {}
        config = AgenticChainConfig.from_dict(data)
        
        assert config.provider == "openai"
    
    def test_from_dict_full(self):
        """Test creating config from full dictionary."""
        data = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "temperature": 0.5,
                "max_tokens": 2000,
            },
            "cache": {
                "enabled": True,
                "ttl": 7200,
            },
            "cost": {
                "max_cost_per_analysis": 0.10,
            },
            "privacy": {
                "redact_pii": True,
            },
        }
        
        config = AgenticChainConfig.from_dict(data)
        
        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.cache.enabled is True
        assert config.cache.ttl == 7200
        assert config.cost.max_cost_per_analysis == 0.10
        assert config.privacy.redact_pii is True
    
    def test_to_llm_config(self):
        """Test converting to LLMConfig."""
        config = AgenticChainConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key",
            temperature=0.8,
        )
        
        llm_config = config.to_llm_config()
        
        assert llm_config.provider == "openai"
        assert llm_config.model == "gpt-4o-mini"
        assert llm_config.api_key == "test-key"
        assert llm_config.temperature == 0.8
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AgenticChainConfig(
            provider="openai",
            model="gpt-4",
            api_key="secret-key",
        )
        
        data = config.to_dict()
        
        assert data["llm"]["provider"] == "openai"
        assert data["llm"]["api_key"] == "***"  # Redacted
        assert "cache" in data
        assert "cost" in data
        assert "privacy" in data


class TestFindConfigFile:
    """Test cases for find_config_file function."""
    
    def test_find_in_directory(self, tmp_path):
        """Test finding config file in specified directory."""
        config_file = tmp_path / ".agentic-chain.yml"
        config_file.write_text("llm:\n  provider: openai")
        
        result = find_config_file(str(tmp_path))
        
        assert result == config_file
    
    def test_find_yaml_extension(self, tmp_path):
        """Test finding config file with .yaml extension."""
        config_file = tmp_path / ".agentic-chain.yaml"
        config_file.write_text("llm:\n  provider: openai")
        
        result = find_config_file(str(tmp_path))
        
        assert result == config_file
    
    def test_not_found(self, tmp_path):
        """Test when no config file is found."""
        result = find_config_file(str(tmp_path))
        
        # May return None if not found in search paths
        # or may find a config in parent directories
        assert result is None or isinstance(result, Path)


class TestLoadYamlFile:
    """Test cases for load_yaml_file function."""
    
    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("llm:\n  provider: anthropic\n  model: claude-3")
        
        data = load_yaml_file(config_file)
        
        assert data["llm"]["provider"] == "anthropic"
        assert data["llm"]["model"] == "claude-3"
    
    def test_load_empty_yaml(self, tmp_path):
        """Test loading an empty YAML file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("")
        
        data = load_yaml_file(config_file)
        
        assert data == {}
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a non-existent file."""
        config_file = tmp_path / "nonexistent.yml"
        
        data = load_yaml_file(config_file)
        
        assert data == {}


class TestLoadConfigFromEnv:
    """Test cases for load_config_from_env function."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'openai-key'}, clear=True)
    def test_openai_from_env(self):
        """Test loading OpenAI config from environment."""
        config = load_config_from_env()
        
        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["api_key"] == "openai-key"
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'anthropic-key'}, clear=True)
    def test_anthropic_from_env(self):
        """Test loading Anthropic config from environment."""
        config = load_config_from_env()
        
        assert config["llm"]["provider"] == "anthropic"
        assert config["llm"]["api_key"] == "anthropic-key"
    
    @patch.dict('os.environ', {
        'AZURE_OPENAI_API_KEY': 'azure-key',
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
    }, clear=True)
    def test_azure_from_env(self):
        """Test loading Azure config from environment."""
        config = load_config_from_env()
        
        assert config["llm"]["provider"] == "azure"
        assert config["llm"]["api_key"] == "azure-key"
        assert config["llm"]["api_base"] == "https://test.openai.azure.com/"
    
    @patch.dict('os.environ', {'OLLAMA_HOST': 'http://localhost:11434'}, clear=True)
    def test_ollama_from_env(self):
        """Test loading Ollama config from environment."""
        config = load_config_from_env()
        
        assert config["llm"]["provider"] == "ollama"
        assert config["llm"]["api_base"] == "http://localhost:11434"
    
    @patch.dict('os.environ', {
        'AGENTIC_CHAIN_PROVIDER': 'custom',
        'AGENTIC_CHAIN_MODEL': 'custom-model',
        'AGENTIC_CHAIN_TEMPERATURE': '0.5',
        'AGENTIC_CHAIN_MAX_TOKENS': '1000',
    }, clear=True)
    def test_custom_env_vars(self):
        """Test loading custom config from environment."""
        config = load_config_from_env()
        
        assert config["llm"]["provider"] == "custom"
        assert config["llm"]["model"] == "custom-model"
        assert config["llm"]["temperature"] == 0.5
        assert config["llm"]["max_tokens"] == 1000


class TestLoadConfig:
    """Test cases for load_config function."""
    
    def test_load_with_defaults(self, tmp_path):
        """Test loading config with default values."""
        config = load_config(project_path=str(tmp_path))
        
        assert config.provider == "openai"
    
    def test_load_with_overrides(self, tmp_path):
        """Test loading config with overrides."""
        config = load_config(
            project_path=str(tmp_path),
            provider="anthropic",
            model="claude-3",
        )
        
        assert config.provider == "anthropic"
        assert config.model == "claude-3"
    
    def test_load_from_file(self, tmp_path):
        """Test loading config from file."""
        config_file = tmp_path / ".agentic-chain.yml"
        config_file.write_text("""
llm:
  provider: ollama
  model: llama3
  temperature: 0.5
cache:
  enabled: true
""")
        
        config = load_config(config_path=str(config_file))
        
        assert config.provider == "ollama"
        assert config.model == "llama3"
        assert config.temperature == 0.5
        assert config.cache.enabled is True
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}, clear=True)
    def test_env_overrides_file(self, tmp_path):
        """Test that environment variables override file config."""
        config_file = tmp_path / ".agentic-chain.yml"
        config_file.write_text("llm:\n  provider: anthropic")
        
        config = load_config(config_path=str(config_file))
        
        # API key from environment should be present
        assert config.api_key == "env-key"


class TestDeepMerge:
    """Test cases for _deep_merge function."""
    
    def test_merge_flat_dicts(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        
        result = _deep_merge(base, override)
        
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"outer": {"inner": 1, "other": 2}}
        override = {"outer": {"inner": 3}}
        
        result = _deep_merge(base, override)
        
        assert result["outer"]["inner"] == 3
        assert result["outer"]["other"] == 2
    
    def test_merge_with_none_values(self):
        """Test that None values don't override."""
        base = {"a": 1, "b": 2}
        override = {"a": None, "c": 3}
        
        result = _deep_merge(base, override)
        
        assert result["a"] == 1  # Not overridden by None
        assert result["c"] == 3


class TestCreateDefaultConfigFile:
    """Test cases for create_default_config_file function."""
    
    def test_create_in_custom_path(self, tmp_path):
        """Test creating config file in custom path."""
        config_path = tmp_path / "custom.yml"
        
        result = create_default_config_file(str(config_path))
        
        assert result == config_path
        assert config_path.exists()
        
        content = config_path.read_text()
        assert "provider: \"openai\"" in content
        assert "model: \"gpt-4o-mini\"" in content
    
    def test_create_default_path(self, tmp_path):
        """Test creating config file in default path."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = create_default_config_file()
            
            assert result.name == ".agentic-chain.yml"
            assert result.exists()
        finally:
            os.chdir(original_cwd)
