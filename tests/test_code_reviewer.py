"""Tests for the CodeReviewer agent."""

import pytest
from pathlib import Path

from agentic_chain.agents import AgentContext
from agentic_chain.agents.code_reviewer import CodeReviewer


class TestCodeReviewer:
    """Test cases for CodeReviewer agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reviewer = CodeReviewer()
    
    def test_init(self):
        """Test reviewer initialization."""
        assert self.reviewer.name == "CodeReviewer"
    
    def test_execute_without_project_analysis(self, tmp_path):
        """Test execution without project analysis raises error."""
        context = AgentContext(project_path=str(tmp_path))
        
        with pytest.raises(ValueError, match="Project analysis is required"):
            self.reviewer.execute(context)
    
    def test_execute_with_valid_context(self, sample_project, sample_issue):
        """Test execution with valid context."""
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {
            "languages": {"Python": 10},
            "structure": {}
        }
        context.issue_data = sample_issue
        context.issue_analysis = {
            "affected_files": ["src/main.py"],
            "keywords": ["subtract", "function"],
            "issue_type": "feature"
        }
        
        result = self.reviewer.execute(context)
        
        assert result.code_review is not None
        assert "relevant_files" in result.code_review
        assert "code_quality" in result.code_review
        assert "potential_issues" in result.code_review
        assert "suggestions" in result.code_review
        assert "file_analyses" in result.code_review
    
    def test_find_relevant_files_from_issue(self, sample_project):
        """Test finding files mentioned in issue."""
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        context.issue_analysis = {
            "affected_files": ["main.py", "test_main.py"],
            "keywords": []
        }
        
        relevant_files = self.reviewer._find_relevant_files(context)
        
        assert any("main.py" in f for f in relevant_files)
    
    def test_find_relevant_files_by_keywords(self, sample_project):
        """Test finding files by keywords."""
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        context.issue_analysis = {
            "affected_files": [],
            "keywords": ["test", "main"]
        }
        
        relevant_files = self.reviewer._find_relevant_files(context)
        
        assert len(relevant_files) > 0
        assert any("test" in f.lower() or "main" in f.lower() for f in relevant_files)
    
    def test_find_relevant_files_fallback(self, sample_project):
        """Test fallback to main source files."""
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        context.issue_analysis = {
            "affected_files": [],
            "keywords": []
        }
        
        relevant_files = self.reviewer._find_relevant_files(context)
        
        assert len(relevant_files) > 0
        assert any(f.endswith('.py') for f in relevant_files)
    
    def test_analyze_code_quality_detects_tests(self, sample_project):
        """Test code quality detects test directory."""
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        
        quality = self.reviewer._analyze_code_quality(context)
        
        assert quality["has_tests"] is True
    
    def test_analyze_code_quality_detects_documentation(self, sample_project):
        """Test code quality detects documentation."""
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        
        quality = self.reviewer._analyze_code_quality(context)
        
        assert quality["has_documentation"] is True
    
    def test_analyze_code_quality_detects_linting_config(self, sample_project):
        """Test code quality detects linting config."""
        (sample_project / ".pylintrc").write_text("[pylint]")
        
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        
        quality = self.reviewer._analyze_code_quality(context)
        
        assert quality["has_linting_config"] is True
    
    def test_identify_potential_issues_finds_todos(self, sample_project):
        """Test identifying TODO comments."""
        (sample_project / "src" / "todos.py").write_text("""
# TODO: Implement this function
def placeholder():
    pass

# FIXME: This is broken
def broken_function():
    return None
""")
        
        context = AgentContext(project_path=str(sample_project))
        context.project_analysis = {"languages": {"Python": 1}}
        
        issues = self.reviewer._identify_potential_issues(context)
        
        assert len(issues) > 0
        assert any(issue["type"] == "TODO" for issue in issues)
        assert any(issue["type"] == "FIXME" for issue in issues)
    
    def test_generate_suggestions_for_missing_tests(self, tmp_path):
        """Test generating suggestions when tests are missing."""
        context = AgentContext(project_path=str(tmp_path))
        context.project_analysis = {"languages": {"Python": 1}}
        context.code_review = {
            "code_quality": {
                "has_tests": False,
                "has_documentation": True,
                "has_type_hints": False,
                "has_linting_config": False
            }
        }
        
        suggestions = self.reviewer._generate_suggestions(context)
        
        assert len(suggestions) > 0
        assert any(s["type"] == "testing" for s in suggestions)
    
    def test_generate_suggestions_for_bug_fix(self, tmp_path):
        """Test generating suggestions for bug fixes."""
        context = AgentContext(project_path=str(tmp_path))
        context.project_analysis = {"languages": {"Python": 1}}
        context.issue_analysis = {"issue_type": "bug"}
        context.code_review = {"code_quality": {}}
        
        suggestions = self.reviewer._generate_suggestions(context)
        
        assert any(s["type"] == "bug_fix" for s in suggestions)
        assert any("regression test" in s["suggestion"].lower() for s in suggestions)
    
    def test_analyze_file_counts_lines(self, sample_project):
        """Test file analysis counts lines correctly."""
        test_file = sample_project / "src" / "main.py"
        
        analysis = self.reviewer._analyze_file(test_file)
        
        assert analysis["lines"] > 0
        assert analysis["blank_lines"] >= 0
        assert analysis["comment_lines"] >= 0
    
    def test_analyze_file_finds_functions(self, sample_project):
        """Test file analysis finds function definitions."""
        test_file = sample_project / "src" / "main.py"
        
        analysis = self.reviewer._analyze_file(test_file)
        
        assert len(analysis["functions"]) > 0
        assert "hello_world" in analysis["functions"]
        assert "add" in analysis["functions"]
    
    def test_analyze_file_finds_imports(self, tmp_path):
        """Test file analysis finds imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import os
from pathlib import Path
import sys
""")
        
        analysis = self.reviewer._analyze_file(test_file)
        
        assert len(analysis["imports"]) >= 3
        assert "os" in analysis["imports"]
    
    def test_analyze_file_handles_errors_gracefully(self, tmp_path):
        """Test file analysis handles errors gracefully."""
        non_existent = tmp_path / "does_not_exist.py"
        
        # Should not raise error
        analysis = self.reviewer._analyze_file(non_existent)
        
        assert analysis["lines"] == 0
