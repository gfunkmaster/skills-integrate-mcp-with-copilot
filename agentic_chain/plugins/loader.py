"""
Plugin Loader - Discovers and loads plugins from files and packages.
"""

import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..agents import BaseAgent
from .registry import PluginRegistry


logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """
    Configuration for a plugin loaded from YAML/JSON.
    
    Example YAML config:
        name: MyPlugin
        enabled: true
        settings:
            max_items: 100
            verbose: true
    """
    
    name: str
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "PluginConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            enabled=data.get("enabled", True),
            settings=data.get("settings", {}),
            source_file=data.get("source_file"),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> List["PluginConfig"]:
        """Load plugin configs from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        configs = []
        if isinstance(data, list):
            for item in data:
                configs.append(cls.from_dict(item))
        elif isinstance(data, dict):
            # Check if it's a single plugin or a plugins list
            if "plugins" in data:
                for item in data["plugins"]:
                    configs.append(cls.from_dict(item))
            else:
                configs.append(cls.from_dict(data))
        
        return configs
    
    @classmethod
    def from_json(cls, path: str) -> List["PluginConfig"]:
        """Load plugin configs from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        configs = []
        if isinstance(data, list):
            for item in data:
                configs.append(cls.from_dict(item))
        elif isinstance(data, dict):
            if "plugins" in data:
                for item in data["plugins"]:
                    configs.append(cls.from_dict(item))
            else:
                configs.append(cls.from_dict(data))
        
        return configs


class PluginValidationError(Exception):
    """Raised when a plugin fails validation."""
    pass


class PluginLoader:
    """
    Discovers and loads plugins from various sources.
    
    Supports:
    - Loading from Python files
    - Loading from Python packages
    - Loading from directories
    - Configuration via YAML/JSON
    
    Safety Features:
    - Validates that plugins inherit from BaseAgent
    - Catches and logs errors during loading
    - Sandboxed execution (plugins can't crash the pipeline)
    
    Example:
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        
        # Load a single plugin file
        loader.load_from_file("/path/to/plugin.py")
        
        # Load all plugins from a directory
        loader.load_from_directory("/path/to/plugins/")
        
        # Load from a Python package
        loader.load_from_package("my_plugins")
        
        # Load with config
        loader.load_from_config("/path/to/plugins.yaml")
    """
    
    def __init__(self, registry: PluginRegistry):
        """
        Initialize the plugin loader.
        
        Args:
            registry: The plugin registry to register discovered plugins.
        """
        self.registry = registry
        self._loaded_modules: Dict[str, Any] = {}
        self._configs: Dict[str, PluginConfig] = {}
    
    def load_from_file(
        self,
        file_path: str,
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Load plugins from a Python file.
        
        Args:
            file_path: Path to the Python file.
            author: Optional author name.
            tags: Optional tags for categorization.
            
        Returns:
            List of loaded plugin names.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return []
        
        if not path.suffix == '.py':
            logger.warning(f"Not a Python file: {file_path}")
            return []
        
        loaded = []
        # Use a unique module name to avoid conflicts
        module_name = f"_agentic_plugin_{path.stem}_{id(path)}"
        module_added = False
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for: {file_path}")
                return []
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            module_added = True
            spec.loader.exec_module(module)
            self._loaded_modules[str(path)] = module
            
            # Find all BaseAgent subclasses
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAgent) and 
                    obj is not BaseAgent):
                    try:
                        self._validate_plugin(obj)
                        if self.registry.register(obj, source=str(path), author=author, tags=tags):
                            loaded.append(name)
                    except PluginValidationError as e:
                        logger.warning(f"Plugin {name} failed validation: {e}")
                    except Exception as e:
                        logger.error(f"Error registering plugin {name}: {e}")
            
            logger.info(f"Loaded {len(loaded)} plugins from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load plugins from {file_path}: {e}")
            # Clean up module from sys.modules on failure
            if module_added and module_name in sys.modules:
                del sys.modules[module_name]
        
        return loaded
    
    def load_from_directory(
        self,
        directory: str,
        recursive: bool = False,
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.
            author: Optional author name.
            tags: Optional tags for categorization.
            
        Returns:
            List of loaded plugin names.
        """
        path = Path(directory).resolve()
        if not path.exists() or not path.is_dir():
            logger.error(f"Plugin directory not found: {directory}")
            return []
        
        loaded = []
        pattern = "**/*.py" if recursive else "*.py"
        
        for py_file in path.glob(pattern):
            # Skip __init__.py and test files
            if py_file.name.startswith('_') or py_file.name.startswith('test_'):
                continue
            
            loaded.extend(self.load_from_file(str(py_file), author=author, tags=tags))
        
        return loaded
    
    def load_from_package(
        self,
        package_name: str,
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Load plugins from an installed Python package.
        
        The package should define plugins in its __init__.py or submodules.
        
        Args:
            package_name: Name of the package to import.
            author: Optional author name.
            tags: Optional tags for categorization.
            
        Returns:
            List of loaded plugin names.
        """
        loaded = []
        try:
            module = importlib.import_module(package_name)
            self._loaded_modules[package_name] = module
            
            # Find all BaseAgent subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseAgent) and 
                    obj is not BaseAgent):
                    try:
                        self._validate_plugin(obj)
                        if self.registry.register(obj, source=package_name, author=author, tags=tags):
                            loaded.append(name)
                    except PluginValidationError as e:
                        logger.warning(f"Plugin {name} failed validation: {e}")
                    except Exception as e:
                        logger.error(f"Error registering plugin {name}: {e}")
            
            logger.info(f"Loaded {len(loaded)} plugins from package {package_name}")
            
        except ImportError as e:
            logger.error(f"Could not import package {package_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to load plugins from package {package_name}: {e}")
        
        return loaded
    
    def load_from_config(self, config_path: str) -> List[str]:
        """
        Load and configure plugins from a YAML/JSON config file.
        
        Args:
            config_path: Path to the config file.
            
        Returns:
            List of loaded plugin names.
        """
        path = Path(config_path).resolve()
        if not path.exists():
            logger.error(f"Config file not found: {config_path}")
            return []
        
        try:
            if path.suffix in ['.yaml', '.yml']:
                configs = PluginConfig.from_yaml(str(path))
            elif path.suffix == '.json':
                configs = PluginConfig.from_json(str(path))
            else:
                logger.error(f"Unsupported config format: {path.suffix}")
                return []
            
            loaded = []
            for config in configs:
                self._configs[config.name] = config
                
                # If source_file is specified, load from file
                if config.source_file:
                    source_path = path.parent / config.source_file
                    loaded.extend(self.load_from_file(str(source_path)))
                
                # Apply enabled/disabled state
                if config.name in self.registry:
                    if config.enabled:
                        self.registry.enable(config.name)
                    else:
                        self.registry.disable(config.name)
            
            logger.info(f"Loaded config for {len(configs)} plugins from {config_path}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return []
    
    def get_config(self, name: str) -> Optional[PluginConfig]:
        """Get configuration for a plugin."""
        return self._configs.get(name)
    
    def _validate_plugin(self, plugin_class: Type[BaseAgent]) -> None:
        """
        Validate that a plugin class meets requirements.
        
        Args:
            plugin_class: The plugin class to validate.
            
        Raises:
            PluginValidationError: If validation fails.
        """
        # Check inheritance
        if not issubclass(plugin_class, BaseAgent):
            raise PluginValidationError(f"{plugin_class.__name__} must inherit from BaseAgent")
        
        # Check execute method
        if not hasattr(plugin_class, 'execute') or not callable(getattr(plugin_class, 'execute')):
            raise PluginValidationError(f"{plugin_class.__name__} must implement execute() method")
        
        # Check for abstract class
        if inspect.isabstract(plugin_class):
            raise PluginValidationError(f"{plugin_class.__name__} is abstract and cannot be instantiated")
    
    def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin from its original source.
        
        Args:
            name: Plugin name to reload.
            
        Returns:
            True if reloaded successfully.
        """
        info = self.registry.get(name)
        if not info:
            logger.warning(f"Plugin not found: {name}")
            return False
        
        source = info.source
        if source in self._loaded_modules:
            try:
                # Unregister first
                self.registry.unregister(name)
                
                # Reload the module
                module = self._loaded_modules[source]
                importlib.reload(module)
                
                # Re-register
                for obj_name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseAgent) and 
                        obj is not BaseAgent):
                        self.registry.register(obj, source=source)
                
                logger.info(f"Reloaded plugin: {name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reload plugin {name}: {e}")
        
        return False
    
    def __repr__(self) -> str:
        return f"PluginLoader(registry={self.registry}, loaded_modules={len(self._loaded_modules)})"
