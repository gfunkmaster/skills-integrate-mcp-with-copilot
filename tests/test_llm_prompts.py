"""Tests for the LLM prompts module."""

import pytest
from agentic_chain.llm.prompts import (
    PROJECT_ANALYSIS_SYSTEM,
    PROJECT_ANALYSIS_PROMPT,
    ISSUE_CLASSIFICATION_SYSTEM,
    ISSUE_CLASSIFICATION_PROMPT,
    CODE_REVIEW_SYSTEM,
    CODE_REVIEW_PROMPT,
    CODE_REVIEW_SUMMARY_PROMPT,
    IMPLEMENTATION_PLAN_SYSTEM,
    IMPLEMENTATION_PLAN_PROMPT,
    format_project_analysis_prompt,
    format_issue_classification_prompt,
    format_code_review_prompt,
    format_implementation_plan_prompt,
)


class TestPromptConstants:
    """Test cases for prompt constants."""
    
    def test_project_analysis_system_not_empty(self):
        """Test that PROJECT_ANALYSIS_SYSTEM is defined."""
        assert PROJECT_ANALYSIS_SYSTEM
        assert "architect" in PROJECT_ANALYSIS_SYSTEM.lower()
    
    def test_project_analysis_prompt_has_placeholders(self):
        """Test that PROJECT_ANALYSIS_PROMPT has expected placeholders."""
        assert "{directory_structure}" in PROJECT_ANALYSIS_PROMPT
        assert "{dependencies}" in PROJECT_ANALYSIS_PROMPT
        assert "{languages}" in PROJECT_ANALYSIS_PROMPT
        assert "{config_files}" in PROJECT_ANALYSIS_PROMPT
    
    def test_issue_classification_system_not_empty(self):
        """Test that ISSUE_CLASSIFICATION_SYSTEM is defined."""
        assert ISSUE_CLASSIFICATION_SYSTEM
        assert "issue" in ISSUE_CLASSIFICATION_SYSTEM.lower()
    
    def test_issue_classification_prompt_has_placeholders(self):
        """Test that ISSUE_CLASSIFICATION_PROMPT has expected placeholders."""
        assert "{issue_title}" in ISSUE_CLASSIFICATION_PROMPT
        assert "{issue_body}" in ISSUE_CLASSIFICATION_PROMPT
        assert "{labels}" in ISSUE_CLASSIFICATION_PROMPT
    
    def test_code_review_system_not_empty(self):
        """Test that CODE_REVIEW_SYSTEM is defined."""
        assert CODE_REVIEW_SYSTEM
        assert "review" in CODE_REVIEW_SYSTEM.lower()
    
    def test_code_review_prompt_has_placeholders(self):
        """Test that CODE_REVIEW_PROMPT has expected placeholders."""
        assert "{code}" in CODE_REVIEW_PROMPT
        assert "{file_path}" in CODE_REVIEW_PROMPT
        assert "{issue_summary}" in CODE_REVIEW_PROMPT
    
    def test_implementation_plan_system_not_empty(self):
        """Test that IMPLEMENTATION_PLAN_SYSTEM is defined."""
        assert IMPLEMENTATION_PLAN_SYSTEM
        assert "implementation" in IMPLEMENTATION_PLAN_SYSTEM.lower()
    
    def test_implementation_plan_prompt_has_placeholders(self):
        """Test that IMPLEMENTATION_PLAN_PROMPT has expected placeholders."""
        assert "{issue_description}" in IMPLEMENTATION_PLAN_PROMPT
        assert "{issue_type}" in IMPLEMENTATION_PLAN_PROMPT
        assert "{priority}" in IMPLEMENTATION_PLAN_PROMPT


class TestFormatProjectAnalysisPrompt:
    """Test cases for format_project_analysis_prompt."""
    
    def test_basic_format(self):
        """Test basic prompt formatting."""
        result = format_project_analysis_prompt(
            directory_structure="- src\n- tests",
            dependencies="pytest, fastapi",
            languages="Python: 10 files",
            config_files="pyproject.toml",
        )
        
        assert "src" in result
        assert "tests" in result
        assert "pytest" in result
        assert "Python" in result
        assert "pyproject.toml" in result
    
    def test_with_readme(self):
        """Test prompt formatting with README."""
        result = format_project_analysis_prompt(
            directory_structure="- src",
            dependencies="none",
            languages="Python: 5",
            config_files="none",
            readme="# My Project\nA sample project.",
        )
        
        assert "README" in result
        assert "My Project" in result
    
    def test_readme_truncation(self):
        """Test that long README is truncated."""
        long_readme = "x" * 3000
        result = format_project_analysis_prompt(
            directory_structure="- src",
            dependencies="none",
            languages="Python: 5",
            config_files="none",
            readme=long_readme,
        )
        
        assert "truncated" in result.lower()


class TestFormatIssueClassificationPrompt:
    """Test cases for format_issue_classification_prompt."""
    
    def test_basic_format(self):
        """Test basic prompt formatting."""
        result = format_issue_classification_prompt(
            issue_title="Fix login bug",
            issue_body="Users cannot login after password reset",
            labels="bug, high-priority",
        )
        
        assert "Fix login bug" in result
        assert "Users cannot login" in result
        assert "bug" in result
    
    def test_with_project_context(self):
        """Test prompt formatting with project context."""
        result = format_issue_classification_prompt(
            issue_title="Add feature",
            issue_body="We need X",
            labels="enhancement",
            project_context="Languages: Python\nFramework: FastAPI",
        )
        
        assert "Python" in result
        assert "FastAPI" in result
    
    def test_empty_body(self):
        """Test handling of empty issue body."""
        result = format_issue_classification_prompt(
            issue_title="Quick fix",
            issue_body="",
        )
        
        assert "No description provided" in result


class TestFormatCodeReviewPrompt:
    """Test cases for format_code_review_prompt."""
    
    def test_basic_format(self):
        """Test basic prompt formatting."""
        result = format_code_review_prompt(
            code="def hello(): print('hello')",
            file_path="src/main.py",
            issue_summary="Fix greeting function",
        )
        
        assert "def hello" in result
        assert "src/main.py" in result
        assert "Fix greeting" in result
    
    def test_with_language(self):
        """Test prompt formatting with language specified."""
        result = format_code_review_prompt(
            code="function hello() { console.log('hi'); }",
            file_path="src/main.js",
            issue_summary="Fix JS code",
            language="javascript",
        )
        
        assert "javascript" in result


class TestFormatImplementationPlanPrompt:
    """Test cases for format_implementation_plan_prompt."""
    
    def test_basic_format(self):
        """Test basic prompt formatting."""
        result = format_implementation_plan_prompt(
            issue_description="Add user authentication",
            issue_type="feature",
            priority="high",
            requirements="OAuth2 support, JWT tokens",
            project_context="Python, FastAPI",
            relevant_files="src/auth.py, src/users.py",
            code_review_summary="Code needs refactoring",
        )
        
        assert "Add user authentication" in result
        assert "feature" in result
        assert "high" in result
        assert "OAuth2" in result
        assert "FastAPI" in result
