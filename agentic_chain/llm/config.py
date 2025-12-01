"""
Configuration Module - Load and manage agentic chain configuration.

This module provides support for loading configuration from:
- YAML configuration files (.agentic-chain.yml)
- Environment variables
- Programmatic configuration

Configuration precedence (highest to lowest):
1. Programmatic configuration (passed to constructor)
2. Environment variables
3. Configuration file
4. Default values
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import LLMConfig


logger = logging.getLogger(__name__)


# Default configuration file names (in order of precedence)
CONFIG_FILE_NAMES = [
    ".agentic-chain.yml",
    ".agentic-chain.yaml",
    "agentic-chain.yml",
    "agentic-chain.yaml",
]


@dataclass
class CacheConfig:
    """Configuration for LLM response caching."""
    
    enabled: bool = False
    ttl: int = 3600  # seconds
    max_size: int = 1000  # maximum cached responses
    storage: str = "memory"  # "memory" or "disk"
    disk_path: Optional[str] = None


@dataclass
class CostConfig:
    """Configuration for cost management."""
    
    max_cost_per_analysis: float = 0.05  # USD
    budget_alert_threshold: float = 10.00  # Monthly budget alert
    track_usage: bool = True


@dataclass
class PrivacyConfig:
    """Configuration for privacy settings."""
    
    redact_pii: bool = False
    share_telemetry: bool = False
    local_only: bool = False  # Force local model usage


@dataclass
class AgenticChainConfig:
    """
    Complete configuration for Agentic Chain.
    
    This configuration can be loaded from:
    - YAML files (.agentic-chain.yml)
    - Environment variables
    - Programmatic configuration
    
    Example YAML configuration:
        ```yaml
        llm:
          provider: "openai"
          model: "gpt-4o-mini"
          temperature: 0.7
          max_tokens: 2000
          
        cache:
          enabled: true
          ttl: 3600
          
        cost:
          max_cost_per_analysis: 0.05
          budget_alert_threshold: 10.00
          
        privacy:
          redact_pii: true
          share_telemetry: false
        ```
    """
    
    # LLM configuration
    provider: str = "openai"
    model: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 3
    
    # Sub-configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    
    # Extra provider-specific options
    extra_options: dict = field(default_factory=dict)
    
    def to_llm_config(self) -> LLMConfig:
        """Convert to LLMConfig for provider initialization."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            extra_options=self.extra_options,
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgenticChainConfig":
        """Create configuration from dictionary."""
        # Extract LLM config
        llm_config = data.get("llm", {})
        
        # Extract sub-configurations
        cache_data = data.get("cache", {})
        cost_data = data.get("cost", {})
        privacy_data = data.get("privacy", {})
        
        return cls(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", ""),
            api_key=llm_config.get("api_key"),
            api_base=llm_config.get("api_base"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 4096),
            timeout=llm_config.get("timeout", 60.0),
            max_retries=llm_config.get("max_retries", 3),
            cache=CacheConfig(
                enabled=cache_data.get("enabled", False),
                ttl=cache_data.get("ttl", 3600),
                max_size=cache_data.get("max_size", 1000),
                storage=cache_data.get("storage", "memory"),
                disk_path=cache_data.get("disk_path"),
            ),
            cost=CostConfig(
                max_cost_per_analysis=cost_data.get("max_cost_per_analysis", 0.05),
                budget_alert_threshold=cost_data.get("budget_alert_threshold", 10.00),
                track_usage=cost_data.get("track_usage", True),
            ),
            privacy=PrivacyConfig(
                redact_pii=privacy_data.get("redact_pii", False),
                share_telemetry=privacy_data.get("share_telemetry", False),
                local_only=privacy_data.get("local_only", False),
            ),
            extra_options=llm_config.get("extra_options", {}),
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "provider": self.provider,
                "model": self.model,
                "api_key": "***" if self.api_key else None,  # Redact API key
                "api_base": self.api_base,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "extra_options": self.extra_options,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "ttl": self.cache.ttl,
                "max_size": self.cache.max_size,
                "storage": self.cache.storage,
                "disk_path": self.cache.disk_path,
            },
            "cost": {
                "max_cost_per_analysis": self.cost.max_cost_per_analysis,
                "budget_alert_threshold": self.cost.budget_alert_threshold,
                "track_usage": self.cost.track_usage,
            },
            "privacy": {
                "redact_pii": self.privacy.redact_pii,
                "share_telemetry": self.privacy.share_telemetry,
                "local_only": self.privacy.local_only,
            },
        }


def find_config_file(start_path: Optional[str] = None) -> Optional[Path]:
    """
    Find the configuration file starting from the given path.
    
    Searches in the following order:
    1. The specified start_path directory
    2. Current working directory
    3. Parent directories up to the root
    4. User home directory
    
    Args:
        start_path: Directory to start searching from.
        
    Returns:
        Path to configuration file if found, None otherwise.
    """
    search_dirs = []
    
    # Start with the specified path
    if start_path:
        search_dirs.append(Path(start_path))
    
    # Add current directory
    search_dirs.append(Path.cwd())
    
    # Add parent directories
    current = Path.cwd()
    while current.parent != current:
        current = current.parent
        search_dirs.append(current)
    
    # Add home directory
    search_dirs.append(Path.home())
    
    # Search for config file
    for directory in search_dirs:
        for config_name in CONFIG_FILE_NAMES:
            config_path = directory / config_name
            if config_path.exists() and config_path.is_file():
                logger.debug(f"Found config file: {config_path}")
                return config_path
    
    return None


def load_yaml_file(file_path: Path) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        Dictionary with configuration data.
    """
    try:
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning(
            "PyYAML not installed. Install with: pip install pyyaml"
        )
        return {}
    except Exception as e:
        logger.warning(f"Error loading config file {file_path}: {e}")
        return {}


def load_config_from_env() -> dict:
    """
    Load configuration from environment variables.
    
    Supported environment variables:
    - AGENTIC_CHAIN_PROVIDER: LLM provider name
    - AGENTIC_CHAIN_MODEL: Model name
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - AZURE_OPENAI_API_KEY: Azure OpenAI API key
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
    - OLLAMA_HOST: Ollama host URL
    
    Returns:
        Dictionary with configuration from environment.
    """
    config = {"llm": {}}
    
    # Provider and model
    if os.environ.get("AGENTIC_CHAIN_PROVIDER"):
        config["llm"]["provider"] = os.environ["AGENTIC_CHAIN_PROVIDER"]
    
    if os.environ.get("AGENTIC_CHAIN_MODEL"):
        config["llm"]["model"] = os.environ["AGENTIC_CHAIN_MODEL"]
    
    # API keys (detect provider from available keys)
    if os.environ.get("OPENAI_API_KEY"):
        config["llm"]["api_key"] = os.environ["OPENAI_API_KEY"]
        if "provider" not in config["llm"]:
            config["llm"]["provider"] = "openai"
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        config["llm"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        if "provider" not in config["llm"]:
            config["llm"]["provider"] = "anthropic"
    
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        config["llm"]["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        if "provider" not in config["llm"]:
            config["llm"]["provider"] = "azure"
    
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        config["llm"]["api_base"] = os.environ["AZURE_OPENAI_ENDPOINT"]
    
    if os.environ.get("OLLAMA_HOST"):
        config["llm"]["api_base"] = os.environ["OLLAMA_HOST"]
        if "provider" not in config["llm"]:
            config["llm"]["provider"] = "ollama"
    
    # Temperature
    if os.environ.get("AGENTIC_CHAIN_TEMPERATURE"):
        try:
            config["llm"]["temperature"] = float(
                os.environ["AGENTIC_CHAIN_TEMPERATURE"]
            )
        except ValueError:
            pass
    
    # Max tokens
    if os.environ.get("AGENTIC_CHAIN_MAX_TOKENS"):
        try:
            config["llm"]["max_tokens"] = int(
                os.environ["AGENTIC_CHAIN_MAX_TOKENS"]
            )
        except ValueError:
            pass
    
    return config


def load_config(
    config_path: Optional[str] = None,
    project_path: Optional[str] = None,
    **overrides: Any,
) -> AgenticChainConfig:
    """
    Load configuration from all sources.
    
    Configuration precedence (highest to lowest):
    1. Overrides passed as keyword arguments
    2. Environment variables
    3. Configuration file
    4. Default values
    
    Args:
        config_path: Optional explicit path to config file.
        project_path: Optional project path to search for config.
        **overrides: Configuration overrides.
        
    Returns:
        Merged AgenticChainConfig.
    """
    # Start with empty config
    merged_config: dict = {}
    
    # Load from config file
    if config_path:
        file_path = Path(config_path)
        if file_path.exists():
            file_config = load_yaml_file(file_path)
            merged_config = _deep_merge(merged_config, file_config)
    else:
        # Try to find config file
        config_file = find_config_file(project_path)
        if config_file:
            file_config = load_yaml_file(config_file)
            merged_config = _deep_merge(merged_config, file_config)
    
    # Load from environment
    env_config = load_config_from_env()
    merged_config = _deep_merge(merged_config, env_config)
    
    # Apply overrides
    if overrides:
        override_config = {"llm": {}}
        for key, value in overrides.items():
            if key in ["provider", "model", "api_key", "api_base", 
                       "temperature", "max_tokens", "timeout", "max_retries"]:
                override_config["llm"][key] = value
            else:
                override_config[key] = value
        merged_config = _deep_merge(merged_config, override_config)
    
    return AgenticChainConfig.from_dict(merged_config)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary.
        override: Dictionary with override values.
        
    Returns:
        Merged dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        elif value is not None:
            result[key] = value
    
    return result


def create_default_config_file(path: Optional[str] = None) -> Path:
    """
    Create a default configuration file.
    
    Args:
        path: Optional path for the config file.
        
    Returns:
        Path to the created config file.
    """
    if path:
        config_path = Path(path)
    else:
        config_path = Path.cwd() / ".agentic-chain.yml"
    
    default_content = """# Agentic Chain Configuration
# See documentation for all options

llm:
  # LLM provider: openai, anthropic, azure, ollama
  provider: "openai"
  
  # Model name (or deployment name for Azure)
  model: "gpt-4o-mini"
  
  # Temperature for response randomness (0.0 - 2.0)
  temperature: 0.7
  
  # Maximum tokens for response
  max_tokens: 2000

# Response caching (reduces API costs)
cache:
  enabled: true
  ttl: 3600  # seconds

# Cost management
cost:
  max_cost_per_analysis: 0.05  # USD per analysis
  budget_alert_threshold: 10.00  # Monthly budget alert

# Privacy settings
privacy:
  redact_pii: false  # Redact personally identifiable information
  share_telemetry: false  # Share anonymous usage data
  local_only: false  # Force use of local models only
"""
    
    config_path.write_text(default_content)
    logger.info(f"Created default config file: {config_path}")
    
    return config_path
