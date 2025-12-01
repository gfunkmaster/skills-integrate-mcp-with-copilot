"""
Plugin system for extending the agentic chain with custom agents.

This module provides:
- PluginRegistry: Central registry for managing plugins
- PluginLoader: Discovers and loads plugins from files and packages
- PluginConfig: Configuration for plugins via YAML/JSON
- Built-in example plugins

Example Usage:
    from agentic_chain.plugins import PluginRegistry, PluginLoader
    
    # Create registry and loader
    registry = PluginRegistry()
    loader = PluginLoader(registry)
    
    # Load plugins from a directory
    loader.load_from_directory("/path/to/plugins")
    
    # Get all registered plugins
    plugins = registry.get_all()
"""

from .registry import PluginRegistry, PluginInfo
from .loader import PluginLoader, PluginConfig
from .examples import (
    SecurityAuditor,
    PerformanceAnalyzer,
    DocumentationChecker,
)

__all__ = [
    # Registry
    "PluginRegistry",
    "PluginInfo",
    # Loader
    "PluginLoader",
    "PluginConfig",
    # Example plugins
    "SecurityAuditor",
    "PerformanceAnalyzer",
    "DocumentationChecker",
]
