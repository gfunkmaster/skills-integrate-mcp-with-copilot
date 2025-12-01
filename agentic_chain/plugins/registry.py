"""
Plugin Registry - Central registry for managing agent plugins.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from ..agents import BaseAgent


logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    
    name: str
    agent_class: Type[BaseAgent]
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source: str = "unknown"
    enabled: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "source": self.source,
            "enabled": self.enabled,
        }


class PluginRegistry:
    """
    Central registry for agent plugins.
    
    The registry manages the lifecycle of plugins, providing:
    - Registration and unregistration
    - Plugin discovery by name or tags
    - Plugin instantiation
    - Dependency validation
    
    Example:
        registry = PluginRegistry()
        registry.register(MyPlugin, source="local")
        
        # Get plugin instance
        plugin = registry.create_instance("MyPlugin")
        
        # List all plugins
        for info in registry.get_all():
            print(f"{info.name}: {info.description}")
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
    
    def register(
        self,
        agent_class: Type[BaseAgent],
        source: str = "manual",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a plugin agent class.
        
        Args:
            agent_class: The agent class to register.
            source: Where the plugin came from (e.g., "local", "package").
            author: Plugin author name.
            tags: Optional tags for categorization.
            
        Returns:
            True if registered successfully, False if already exists.
        """
        # Get metadata from class attributes or defaults
        # Use class-level inspection to avoid side effects from instantiation
        name = getattr(agent_class, '__name__', 'UnknownPlugin')
        
        # Check for class-level metadata attributes first
        version = "1.0.0"
        description = ""
        dependencies = []
        
        # Try to get metadata by creating an instance
        # This is safe because all our agents have simple __init__ methods
        try:
            temp_instance = agent_class()
            name = temp_instance.name
            version = temp_instance.version
            description = temp_instance.description
            dependencies = list(temp_instance.dependencies)
        except Exception as e:
            logger.debug(f"Could not get metadata from instance: {e}")
        
        if name in self._plugins:
            logger.warning(f"Plugin {name} already registered")
            return False
        
        info = PluginInfo(
            name=name,
            agent_class=agent_class,
            version=version,
            description=description,
            author=author,
            dependencies=list(dependencies),
            tags=tags or [],
            source=source,
            enabled=True,
        )
        
        self._plugins[name] = info
        logger.info(f"Registered plugin: {name} (v{version})")
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            name: Plugin name to unregister.
            
        Returns:
            True if unregistered, False if not found.
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[PluginInfo]:
        """
        Get plugin info by name.
        
        Args:
            name: Plugin name.
            
        Returns:
            PluginInfo if found, None otherwise.
        """
        return self._plugins.get(name)
    
    def get_all(self) -> List[PluginInfo]:
        """
        Get all registered plugins.
        
        Returns:
            List of PluginInfo for all plugins.
        """
        return list(self._plugins.values())
    
    def get_enabled(self) -> List[PluginInfo]:
        """
        Get all enabled plugins.
        
        Returns:
            List of PluginInfo for enabled plugins.
        """
        return [p for p in self._plugins.values() if p.enabled]
    
    def get_by_tag(self, tag: str) -> List[PluginInfo]:
        """
        Get plugins by tag.
        
        Args:
            tag: Tag to filter by.
            
        Returns:
            List of plugins with the given tag.
        """
        return [p for p in self._plugins.values() if tag in p.tags]
    
    def create_instance(self, name: str, **kwargs) -> Optional[BaseAgent]:
        """
        Create an instance of a registered plugin.
        
        Args:
            name: Plugin name.
            **kwargs: Arguments to pass to the plugin constructor.
            
        Returns:
            Agent instance if found, None otherwise.
        """
        info = self._plugins.get(name)
        if not info:
            logger.warning(f"Plugin not found: {name}")
            return None
        
        if not info.enabled:
            logger.warning(f"Plugin {name} is disabled")
            return None
        
        try:
            return info.agent_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create instance of {name}: {e}")
            return None
    
    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = False
            return True
        return False
    
    def is_registered(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        logger.info("Cleared all plugins from registry")
    
    def validate_dependencies(self, name: str) -> List[str]:
        """
        Check if all dependencies of a plugin are registered.
        
        Args:
            name: Plugin name.
            
        Returns:
            List of missing dependency names.
        """
        info = self._plugins.get(name)
        if not info:
            return [name]  # Plugin itself not found
        
        missing = []
        for dep in info.dependencies:
            if dep not in self._plugins:
                missing.append(dep)
        return missing
    
    def __len__(self) -> int:
        return len(self._plugins)
    
    def __contains__(self, name: str) -> bool:
        return name in self._plugins
    
    def __repr__(self) -> str:
        return f"PluginRegistry(plugins={list(self._plugins.keys())})"
