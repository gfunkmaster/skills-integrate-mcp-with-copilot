# Plugin Development Guide

This guide explains how to create custom agent plugins for the Agentic Chain framework.

## Overview

The plugin system allows you to extend Agentic Chain with custom agents that can:
- Analyze code for specific patterns (security, performance, style)
- Integrate with external tools and services
- Add domain-specific analysis capabilities
- Share reusable functionality across projects

## Quick Start

### 1. Create Your Plugin

Create a Python file with your plugin class:

```python
from agentic_chain.agents import BaseAgent, AgentContext

class MyCustomAgent(BaseAgent):
    """A custom agent that does something useful."""
    
    def __init__(self, name: str = "MyCustomAgent"):
        super().__init__(name)
    
    @property
    def dependencies(self):
        """Optional: specify which agents must run before this one."""
        return ["ProjectAnalyzer"]  # Requires project analysis first
    
    @property
    def description(self):
        """Human-readable description of what this agent does."""
        return "Performs custom analysis on the project"
    
    @property
    def version(self):
        """Version of this plugin."""
        return "1.0.0"
    
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute the agent's task.
        
        Args:
            context: The shared context object with project information.
            
        Returns:
            Updated context with results in plugin_results.
        """
        # Access project information
        project_path = context.project_path
        
        # Access results from other agents
        project_analysis = context.project_analysis
        
        # Perform your custom analysis
        results = self._analyze(project_path)
        
        # Store results in plugin_results (recommended)
        context.plugin_results["my_custom_analysis"] = results
        
        return context
    
    def _analyze(self, project_path: str) -> dict:
        """Your custom analysis logic."""
        return {"status": "completed", "findings": []}
```

### 2. Register Your Plugin

```python
from agentic_chain.plugins import PluginRegistry
from my_plugins import MyCustomAgent

registry = PluginRegistry()
registry.register(MyCustomAgent, source="local", tags=["custom"])
```

### 3. Use Your Plugin

```python
from agentic_chain import AgenticChain

# Create chain with your custom agent
chain = AgenticChain(project_path="/path/to/project")
chain.add_agent(registry.create_instance("MyCustomAgent"))

# Or use with the registry
for info in registry.get_all():
    plugin = registry.create_instance(info.name)
    context = plugin.execute(context)
```

## The BaseAgent Interface

All plugins must inherit from `BaseAgent` and implement the `execute` method:

```python
class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def dependencies(self) -> List[str]:
        """List of agent names that must run before this agent."""
        return []
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return f"Agent: {self.name}"
    
    @property
    def version(self) -> str:
        """Version string."""
        return "1.0.0"
    
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentContext:
        """Execute the agent's task."""
        pass
    
    def validate_context(self, context: AgentContext) -> bool:
        """Validate that context has required data."""
        return context.project_path is not None
```

## The AgentContext

The `AgentContext` is the shared data structure passed between agents:

```python
@dataclass
class AgentContext:
    project_path: str                    # Path to the project
    issue_data: Optional[dict]           # Issue information (if analyzing an issue)
    project_analysis: Optional[dict]     # Results from ProjectAnalyzer
    issue_analysis: Optional[dict]       # Results from IssueAnalyzer
    code_review: Optional[dict]          # Results from CodeReviewer
    solution: Optional[dict]             # Results from SolutionImplementer
    metadata: dict                       # General metadata
    llm_context: LLMContext             # LLM usage tracking
    plugin_results: dict                 # Store plugin-specific results here
```

### Best Practices for Context Usage

1. **Store results in `plugin_results`**: Use a unique key for your plugin
   ```python
   context.plugin_results["my_plugin_name"] = {
       "findings": [...],
       "summary": "...",
   }
   ```

2. **Check for required data**: Verify dependencies are present
   ```python
   if not context.project_analysis:
       raise ValueError("ProjectAnalyzer must run first")
   ```

3. **Don't modify other agents' data**: Only add to `plugin_results` or `metadata`

## Loading Plugins

### From Files

```python
from agentic_chain.plugins import PluginLoader, PluginRegistry

registry = PluginRegistry()
loader = PluginLoader(registry)

# Load from a single file
loader.load_from_file("/path/to/my_plugin.py")

# Load from a directory
loader.load_from_directory("/path/to/plugins/")
```

### From Packages

```python
# Load from an installed Python package
loader.load_from_package("my_plugins_package")
```

### From Configuration Files

Create a YAML configuration file (`plugins.yaml`):

```yaml
plugins:
  - name: SecurityAuditor
    enabled: true
    settings:
      severity_threshold: high
      
  - name: PerformanceAnalyzer
    enabled: true
    
  - name: MyCustomPlugin
    enabled: false  # Disabled
    source_file: custom/my_plugin.py
```

Or JSON (`plugins.json`):

```json
{
  "plugins": [
    {
      "name": "SecurityAuditor",
      "enabled": true,
      "settings": {
        "severity_threshold": "high"
      }
    }
  ]
}
```

Load the configuration:

```python
loader.load_from_config("/path/to/plugins.yaml")
```

## Plugin Registry API

```python
# Register a plugin
registry.register(MyPlugin, source="local", author="Me", tags=["custom"])

# Get plugin info
info = registry.get("MyPlugin")
print(f"Name: {info.name}, Version: {info.version}")

# List all plugins
for info in registry.get_all():
    print(f"{info.name}: {info.description}")

# Filter by tag
security_plugins = registry.get_by_tag("security")

# Create instances
plugin = registry.create_instance("MyPlugin")

# Enable/disable plugins
registry.disable("MyPlugin")
registry.enable("MyPlugin")

# Check registration
if registry.is_registered("MyPlugin"):
    print("Plugin is registered")

# Validate dependencies
missing = registry.validate_dependencies("MyPlugin")
if missing:
    print(f"Missing dependencies: {missing}")
```

## Example Plugins

The framework includes three example plugins you can use as templates:

### SecurityAuditor

Scans code for security vulnerabilities:
- Hardcoded secrets and credentials
- SQL injection vulnerabilities
- Command injection risks
- Unsafe deserialization

```python
from agentic_chain.plugins import SecurityAuditor

auditor = SecurityAuditor()
context = auditor.execute(context)

# Access results
audit = context.plugin_results["security_audit"]
print(f"Risk score: {audit['risk_score']}/100")
print(f"Total findings: {audit['total_findings']}")
```

### PerformanceAnalyzer

Identifies performance bottlenecks:
- Inefficient loop patterns
- N+1 query patterns
- Memory-inefficient code
- Blocking I/O in async code

```python
from agentic_chain.plugins import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
context = analyzer.execute(context)

# Access results
analysis = context.plugin_results["performance_analysis"]
for hotspot in analysis["hotspots"]:
    print(f"Hotspot: {hotspot['file']} ({hotspot['issue_count']} issues)")
```

### DocumentationChecker

Ensures code is properly documented:
- Missing docstrings
- README quality
- Type hints coverage
- Documentation completeness

```python
from agentic_chain.plugins import DocumentationChecker

checker = DocumentationChecker()
context = checker.execute(context)

# Access results
check = context.plugin_results["documentation_check"]
print(f"Documentation score: {check['documentation_score']}/100 ({check['grade']})")
```

## Error Handling

Plugins should handle errors gracefully to avoid crashing the pipeline:

```python
class RobustPlugin(BaseAgent):
    def execute(self, context: AgentContext) -> AgentContext:
        try:
            results = self._risky_operation()
            context.plugin_results["my_plugin"] = {
                "status": "success",
                "results": results,
            }
        except Exception as e:
            # Log the error but don't crash
            context.plugin_results["my_plugin"] = {
                "status": "error",
                "error": str(e),
            }
        return context
```

## Testing Your Plugin

```python
import pytest
from pathlib import Path
from agentic_chain.agents import AgentContext
from my_plugins import MyCustomAgent

def test_my_plugin(tmp_path):
    # Create test project structure
    (tmp_path / "test.py").write_text("print('hello')")
    
    # Create context
    context = AgentContext(project_path=str(tmp_path))
    
    # Execute plugin
    agent = MyCustomAgent()
    result = agent.execute(context)
    
    # Verify results
    assert "my_custom_analysis" in result.plugin_results
    assert result.plugin_results["my_custom_analysis"]["status"] == "completed"
```

## Advanced Topics

### Using LLM Integration

If your plugin needs LLM capabilities:

```python
class LLMPoweredPlugin(BaseAgent):
    def __init__(self, llm_provider=None):
        super().__init__("LLMPoweredPlugin")
        self._llm_provider = llm_provider
    
    @property
    def llm_provider(self):
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider):
        self._llm_provider = provider
    
    def execute(self, context: AgentContext) -> AgentContext:
        if self._llm_provider:
            # Use LLM for analysis
            response = self._llm_provider.complete([...])
            # Track usage
            context.llm_context.add_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        return context
```

### Publishing Your Plugin

1. Create a Python package with your plugins
2. Add `agentic-chain` as a dependency
3. Document your plugins
4. Publish to PyPI

```python
# setup.py or pyproject.toml
[project]
name = "my-agentic-chain-plugins"
dependencies = ["agentic-chain>=0.1.0"]
```

## Summary

Creating plugins for Agentic Chain is straightforward:

1. Inherit from `BaseAgent`
2. Implement `execute(context) -> context`
3. Store results in `context.plugin_results`
4. Register with `PluginRegistry`
5. Test thoroughly

The plugin system ensures:
- Consistent interface across all agents
- Safe error handling
- Easy discovery and loading
- Configuration via YAML/JSON
- Proper dependency management
