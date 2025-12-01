"""Tests for the plugin system."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agentic_chain.agents import BaseAgent, AgentContext
from agentic_chain.plugins import (
    PluginRegistry,
    PluginInfo,
    PluginLoader,
    PluginConfig,
    SecurityAuditor,
    PerformanceAnalyzer,
    DocumentationChecker,
)


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""
    
    def test_to_dict(self):
        """Test converting PluginInfo to dictionary."""
        info = PluginInfo(
            name="TestPlugin",
            agent_class=SecurityAuditor,
            version="1.0.0",
            description="Test description",
            author="Test Author",
            dependencies=["ProjectAnalyzer"],
            tags=["test"],
            source="local",
            enabled=True,
        )
        
        result = info.to_dict()
        
        assert result["name"] == "TestPlugin"
        assert result["version"] == "1.0.0"
        assert result["description"] == "Test description"
        assert result["author"] == "Test Author"
        assert result["dependencies"] == ["ProjectAnalyzer"]
        assert result["tags"] == ["test"]
        assert result["source"] == "local"
        assert result["enabled"] is True


class TestPluginRegistry:
    """Tests for PluginRegistry."""
    
    def test_init(self):
        """Test registry initialization."""
        registry = PluginRegistry()
        
        assert len(registry) == 0
        assert list(registry.get_all()) == []
    
    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()
        
        result = registry.register(SecurityAuditor, source="test")
        
        assert result is True
        assert len(registry) == 1
        assert "SecurityAuditor" in registry
    
    def test_register_duplicate(self):
        """Test registering a duplicate plugin."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        
        result = registry.register(SecurityAuditor)
        
        assert result is False
        assert len(registry) == 1
    
    def test_unregister(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        
        result = registry.unregister("SecurityAuditor")
        
        assert result is True
        assert len(registry) == 0
    
    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent plugin."""
        registry = PluginRegistry()
        
        result = registry.unregister("NonexistentPlugin")
        
        assert result is False
    
    def test_get_plugin(self):
        """Test getting plugin info."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor, source="test", author="Test Author")
        
        info = registry.get("SecurityAuditor")
        
        assert info is not None
        assert info.name == "SecurityAuditor"
        assert info.source == "test"
        assert info.author == "Test Author"
    
    def test_get_nonexistent(self):
        """Test getting nonexistent plugin."""
        registry = PluginRegistry()
        
        info = registry.get("NonexistentPlugin")
        
        assert info is None
    
    def test_get_all(self):
        """Test getting all plugins."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        registry.register(PerformanceAnalyzer)
        registry.register(DocumentationChecker)
        
        plugins = registry.get_all()
        
        assert len(plugins) == 3
        names = [p.name for p in plugins]
        assert "SecurityAuditor" in names
        assert "PerformanceAnalyzer" in names
        assert "DocumentationChecker" in names
    
    def test_get_enabled(self):
        """Test getting enabled plugins."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        registry.register(PerformanceAnalyzer)
        registry.disable("PerformanceAnalyzer")
        
        enabled = registry.get_enabled()
        
        assert len(enabled) == 1
        assert enabled[0].name == "SecurityAuditor"
    
    def test_get_by_tag(self):
        """Test getting plugins by tag."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor, tags=["security", "audit"])
        registry.register(PerformanceAnalyzer, tags=["performance"])
        
        security_plugins = registry.get_by_tag("security")
        
        assert len(security_plugins) == 1
        assert security_plugins[0].name == "SecurityAuditor"
    
    def test_create_instance(self):
        """Test creating plugin instance."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        
        instance = registry.create_instance("SecurityAuditor")
        
        assert instance is not None
        assert isinstance(instance, SecurityAuditor)
        assert instance.name == "SecurityAuditor"
    
    def test_create_instance_disabled(self):
        """Test creating instance of disabled plugin."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        registry.disable("SecurityAuditor")
        
        instance = registry.create_instance("SecurityAuditor")
        
        assert instance is None
    
    def test_create_instance_nonexistent(self):
        """Test creating instance of nonexistent plugin."""
        registry = PluginRegistry()
        
        instance = registry.create_instance("NonexistentPlugin")
        
        assert instance is None
    
    def test_enable_disable(self):
        """Test enabling and disabling plugins."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        
        assert registry.get("SecurityAuditor").enabled is True
        
        registry.disable("SecurityAuditor")
        assert registry.get("SecurityAuditor").enabled is False
        
        registry.enable("SecurityAuditor")
        assert registry.get("SecurityAuditor").enabled is True
    
    def test_is_registered(self):
        """Test checking if plugin is registered."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        
        assert registry.is_registered("SecurityAuditor") is True
        assert registry.is_registered("NonexistentPlugin") is False
    
    def test_clear(self):
        """Test clearing all plugins."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        registry.register(PerformanceAnalyzer)
        
        registry.clear()
        
        assert len(registry) == 0
    
    def test_validate_dependencies(self):
        """Test validating plugin dependencies."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)  # Has ProjectAnalyzer dependency
        
        # ProjectAnalyzer not registered, so it should be missing
        missing = registry.validate_dependencies("SecurityAuditor")
        
        assert "ProjectAnalyzer" in missing
    
    def test_contains(self):
        """Test __contains__ method."""
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        
        assert "SecurityAuditor" in registry
        assert "NonexistentPlugin" not in registry


class TestPluginLoader:
    """Tests for PluginLoader."""
    
    def test_init(self):
        """Test loader initialization."""
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        
        assert loader.registry is registry
    
    def test_load_from_file(self, tmp_path):
        """Test loading plugin from file."""
        # Create a test plugin file
        plugin_code = '''
from agentic_chain.agents import BaseAgent, AgentContext

class TestFilePlugin(BaseAgent):
    """A test plugin loaded from file."""
    
    def __init__(self):
        super().__init__("TestFilePlugin")
    
    @property
    def description(self):
        return "Test plugin from file"
    
    def execute(self, context: AgentContext) -> AgentContext:
        context.plugin_results["test_file"] = {"loaded": True}
        return context
'''
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(plugin_code)
        
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        
        loaded = loader.load_from_file(str(plugin_file))
        
        assert len(loaded) == 1
        assert "TestFilePlugin" in loaded
        assert "TestFilePlugin" in registry
    
    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file."""
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        
        loaded = loader.load_from_file("/nonexistent/path/plugin.py")
        
        assert loaded == []
    
    def test_load_from_directory(self, tmp_path):
        """Test loading plugins from directory."""
        # Create test plugin files
        plugin1 = '''
from agentic_chain.agents import BaseAgent, AgentContext

class Plugin1(BaseAgent):
    def __init__(self):
        super().__init__("Plugin1")
    
    def execute(self, context):
        return context
'''
        plugin2 = '''
from agentic_chain.agents import BaseAgent, AgentContext

class Plugin2(BaseAgent):
    def __init__(self):
        super().__init__("Plugin2")
    
    def execute(self, context):
        return context
'''
        (tmp_path / "plugin1.py").write_text(plugin1)
        (tmp_path / "plugin2.py").write_text(plugin2)
        
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        
        loaded = loader.load_from_directory(str(tmp_path))
        
        assert len(loaded) == 2
        assert "Plugin1" in registry
        assert "Plugin2" in registry
    
    def test_load_from_directory_skips_init(self, tmp_path):
        """Test that loader skips __init__.py files."""
        init_content = '''
from agentic_chain.agents import BaseAgent, AgentContext

class InitPlugin(BaseAgent):
    def __init__(self):
        super().__init__("InitPlugin")
    
    def execute(self, context):
        return context
'''
        (tmp_path / "__init__.py").write_text(init_content)
        
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        
        loaded = loader.load_from_directory(str(tmp_path))
        
        assert loaded == []


class TestPluginConfig:
    """Tests for PluginConfig."""
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "TestPlugin",
            "enabled": True,
            "settings": {"key": "value"},
        }
        
        config = PluginConfig.from_dict(data)
        
        assert config.name == "TestPlugin"
        assert config.enabled is True
        assert config.settings == {"key": "value"}
    
    def test_from_json(self, tmp_path):
        """Test loading configs from JSON file."""
        config_data = {
            "plugins": [
                {"name": "Plugin1", "enabled": True},
                {"name": "Plugin2", "enabled": False},
            ]
        }
        config_file = tmp_path / "plugins.json"
        config_file.write_text(json.dumps(config_data))
        
        configs = PluginConfig.from_json(str(config_file))
        
        assert len(configs) == 2
        assert configs[0].name == "Plugin1"
        assert configs[0].enabled is True
        assert configs[1].name == "Plugin2"
        assert configs[1].enabled is False


class TestSecurityAuditor:
    """Tests for SecurityAuditor plugin."""
    
    def test_init(self):
        """Test initialization."""
        auditor = SecurityAuditor()
        
        assert auditor.name == "SecurityAuditor"
        assert "ProjectAnalyzer" in auditor.dependencies
        assert auditor.version == "1.0.0"
    
    def test_execute(self, tmp_path):
        """Test executing security audit."""
        # Create a test file with security issues
        test_code = '''
password = "secret123"
api_key = "sk-test-key"
def unsafe_query(user_input):
    cursor.execute("SELECT * FROM users WHERE id=" + user_input)
'''
        (tmp_path / "test.py").write_text(test_code)
        
        context = AgentContext(project_path=str(tmp_path))
        auditor = SecurityAuditor()
        
        result = auditor.execute(context)
        
        assert "security_audit" in result.plugin_results
        audit = result.plugin_results["security_audit"]
        assert audit["total_findings"] > 0
        assert audit["risk_score"] > 0
    
    def test_clean_project(self, tmp_path):
        """Test auditing a clean project."""
        clean_code = '''
def add(a, b):
    return a + b
'''
        (tmp_path / "clean.py").write_text(clean_code)
        
        context = AgentContext(project_path=str(tmp_path))
        auditor = SecurityAuditor()
        
        result = auditor.execute(context)
        
        audit = result.plugin_results["security_audit"]
        assert audit["total_findings"] == 0
        assert audit["risk_level"] == "low"


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer plugin."""
    
    def test_init(self):
        """Test initialization."""
        analyzer = PerformanceAnalyzer()
        
        assert analyzer.name == "PerformanceAnalyzer"
        assert "ProjectAnalyzer" in analyzer.dependencies
    
    def test_execute(self, tmp_path):
        """Test executing performance analysis."""
        # Create a test file with performance issues
        test_code = '''
def inefficient():
    result = []
    for i in range(1000):
        result.append(i * 2)  # Should use list comprehension
    return result
'''
        (tmp_path / "test.py").write_text(test_code)
        
        context = AgentContext(project_path=str(tmp_path))
        analyzer = PerformanceAnalyzer()
        
        result = analyzer.execute(context)
        
        assert "performance_analysis" in result.plugin_results
        analysis = result.plugin_results["performance_analysis"]
        assert "complexity_metrics" in analysis


class TestDocumentationChecker:
    """Tests for DocumentationChecker plugin."""
    
    def test_init(self):
        """Test initialization."""
        checker = DocumentationChecker()
        
        assert checker.name == "DocumentationChecker"
        assert "ProjectAnalyzer" in checker.dependencies
    
    def test_execute_with_documented_code(self, tmp_path):
        """Test checking well-documented code."""
        documented_code = '''"""Module docstring."""

class MyClass:
    """Class docstring."""
    
    def my_method(self, x: int) -> int:
        """Method docstring."""
        return x * 2
'''
        (tmp_path / "documented.py").write_text(documented_code)
        (tmp_path / "README.md").write_text("# Project\\n\\nDescription here. " * 20)
        
        context = AgentContext(project_path=str(tmp_path))
        checker = DocumentationChecker()
        
        result = checker.execute(context)
        
        assert "documentation_check" in result.plugin_results
        check = result.plugin_results["documentation_check"]
        assert check["documentation_score"] >= 50
    
    def test_execute_without_readme(self, tmp_path):
        """Test checking code without README."""
        (tmp_path / "test.py").write_text("x = 1")
        
        context = AgentContext(project_path=str(tmp_path))
        checker = DocumentationChecker()
        
        result = checker.execute(context)
        
        check = result.plugin_results["documentation_check"]
        # Should have findings about missing README
        assert any(
            f.get("type") == "missing_readme" 
            for f in check.get("findings", [])
        )


class TestPluginIntegration:
    """Integration tests for the plugin system."""
    
    def test_register_and_execute_all_example_plugins(self, tmp_path):
        """Test registering and executing all example plugins."""
        # Create a sample project
        (tmp_path / "app.py").write_text('"""App module."""\ndef main(): pass')
        (tmp_path / "README.md").write_text("# Test Project\n\n## Installation\n\n## Usage")
        
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        registry.register(PerformanceAnalyzer)
        registry.register(DocumentationChecker)
        
        context = AgentContext(project_path=str(tmp_path))
        
        for info in registry.get_all():
            plugin = registry.create_instance(info.name)
            context = plugin.execute(context)
        
        assert "security_audit" in context.plugin_results
        assert "performance_analysis" in context.plugin_results
        assert "documentation_check" in context.plugin_results
    
    def test_plugin_errors_dont_crash_pipeline(self, tmp_path):
        """Test that plugin errors are handled gracefully."""
        # Create a plugin that raises an error
        error_plugin_code = '''
from agentic_chain.agents import BaseAgent, AgentContext

class ErrorPlugin(BaseAgent):
    def __init__(self):
        super().__init__("ErrorPlugin")
    
    def execute(self, context):
        raise RuntimeError("Intentional error for testing")
'''
        error_file = tmp_path / "error_plugin.py"
        error_file.write_text(error_plugin_code)
        
        registry = PluginRegistry()
        loader = PluginLoader(registry)
        loader.load_from_file(str(error_file))
        
        context = AgentContext(project_path=str(tmp_path))
        plugin = registry.create_instance("ErrorPlugin")
        
        # Should raise an error
        with pytest.raises(RuntimeError):
            plugin.execute(context)
    
    def test_plugin_context_isolation(self, tmp_path):
        """Test that plugins properly isolate their results."""
        (tmp_path / "test.py").write_text("x = 1")
        
        registry = PluginRegistry()
        registry.register(SecurityAuditor)
        registry.register(PerformanceAnalyzer)
        
        context = AgentContext(project_path=str(tmp_path))
        
        # Execute first plugin
        plugin1 = registry.create_instance("SecurityAuditor")
        context = plugin1.execute(context)
        
        # Execute second plugin
        plugin2 = registry.create_instance("PerformanceAnalyzer")
        context = plugin2.execute(context)
        
        # Both results should be present and independent
        assert "security_audit" in context.plugin_results
        assert "performance_analysis" in context.plugin_results
        assert context.plugin_results["security_audit"] != context.plugin_results["performance_analysis"]
