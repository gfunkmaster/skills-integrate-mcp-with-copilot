"""Tests for the ProjectAnalyzer agent."""

import os
import tempfile
import pytest
from pathlib import Path

from agentic_chain.agents import AgentContext
from agentic_chain.agents.project_analyzer import ProjectAnalyzer


class TestProjectAnalyzer:
    """Test cases for ProjectAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ProjectAnalyzer()
        
    def test_init(self):
        """Test analyzer initialization."""
        assert self.analyzer.name == "ProjectAnalyzer"
        
    def test_execute_with_valid_project(self, tmp_path):
        """Test execution with a valid project directory."""
        # Create a simple project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def hello(): pass")
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n")
        (tmp_path / "README.md").write_text("# Test Project")
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        assert result.project_analysis is not None
        assert "structure" in result.project_analysis
        assert "dependencies" in result.project_analysis
        assert "languages" in result.project_analysis
        
    def test_detect_languages(self, tmp_path):
        """Test language detection."""
        # Create files with different extensions
        (tmp_path / "app.py").write_text("# Python file")
        (tmp_path / "index.js").write_text("// JavaScript file")
        (tmp_path / "main.go").write_text("// Go file")
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        languages = result.project_analysis.get("languages", {})
        assert "Python" in languages
        assert "JavaScript" in languages
        assert "Go" in languages
        
    def test_analyze_dependencies_python(self, tmp_path):
        """Test Python dependency analysis."""
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn>=0.20.0\npydantic")
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        deps = result.project_analysis.get("dependencies", {})
        assert deps.get("python") is not None
        assert "fastapi" in deps["python"]
        
    def test_analyze_dependencies_javascript(self, tmp_path):
        """Test JavaScript dependency analysis."""
        import json
        package_json = {
            "name": "test-project",
            "dependencies": {
                "react": "^18.0.0"
            },
            "devDependencies": {
                "jest": "^29.0.0"
            }
        }
        (tmp_path / "package.json").write_text(json.dumps(package_json))
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        deps = result.project_analysis.get("dependencies", {})
        assert deps.get("javascript") is not None
        assert "react" in deps["javascript"]["dependencies"]
        
    def test_detect_patterns_fastapi(self, tmp_path):
        """Test FastAPI framework detection."""
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn")
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        patterns = result.project_analysis.get("patterns", {})
        assert patterns.get("framework") == "FastAPI"
        
    def test_detect_patterns_github_actions(self, tmp_path):
        """Test GitHub Actions CI/CD detection."""
        workflows = tmp_path / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "ci.yml").write_text("name: CI")
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        patterns = result.project_analysis.get("patterns", {})
        assert "GitHub Actions" in patterns.get("ci_cd", [])
        
    def test_read_readme(self, tmp_path):
        """Test README content reading."""
        readme_content = "# My Project\n\nThis is a test project."
        (tmp_path / "README.md").write_text(readme_content)
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        assert result.project_analysis.get("readme") == readme_content
        
    def test_find_config_files(self, tmp_path):
        """Test configuration file detection."""
        (tmp_path / "pyproject.toml").write_text("[project]")
        (tmp_path / "setup.cfg").write_text("[metadata]")
        
        context = AgentContext(project_path=str(tmp_path))
        result = self.analyzer.execute(context)
        
        config_files = result.project_analysis.get("config_files", [])
        assert "pyproject.toml" in config_files
        
    def test_validate_context_missing_path(self):
        """Test context validation with missing project path."""
        context = AgentContext(project_path="")
        # Empty path should be falsy
        assert not self.analyzer.validate_context(AgentContext(project_path=None))
