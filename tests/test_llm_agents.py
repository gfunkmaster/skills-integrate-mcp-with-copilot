"""Tests for LLM-enhanced agents (ProjectAnalyzer, IssueAnalyzer, CodeReviewer)."""

import pytest
from unittest.mock import MagicMock, patch

from agentic_chain.agents import AgentContext
from agentic_chain.agents.project_analyzer import ProjectAnalyzer
from agentic_chain.agents.issue_analyzer import IssueAnalyzer
from agentic_chain.agents.code_reviewer import CodeReviewer
from agentic_chain.llm.base import (
    LLMConfig,
    LLMResponse,
    LLMUsage,
    LLMMessage,
    MessageRole,
)


def create_mock_llm_provider():
    """Create a mock LLM provider for testing."""
    mock_provider = MagicMock()
    mock_provider.config = LLMConfig(provider="openai", model="gpt-4")
    
    # Mock response
    mock_response = LLMResponse(
        content="Mock LLM response content",
        model="gpt-4",
        usage=LLMUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.01,
        ),
        finish_reason="stop",
    )
    mock_provider.complete.return_value = mock_response
    
    return mock_provider


class TestProjectAnalyzerWithLLM:
    """Test cases for ProjectAnalyzer with LLM integration."""
    
    def test_init_without_llm(self):
        """Test initialization without LLM provider."""
        analyzer = ProjectAnalyzer()
        assert analyzer.llm_provider is None
    
    def test_init_with_llm(self):
        """Test initialization with LLM provider."""
        mock_provider = create_mock_llm_provider()
        analyzer = ProjectAnalyzer(llm_provider=mock_provider)
        assert analyzer.llm_provider is mock_provider
    
    def test_set_llm_provider(self):
        """Test setting LLM provider after initialization."""
        analyzer = ProjectAnalyzer()
        mock_provider = create_mock_llm_provider()
        analyzer.llm_provider = mock_provider
        assert analyzer.llm_provider is mock_provider
    
    def test_execute_without_llm(self, sample_project):
        """Test execute without LLM returns static analysis only."""
        analyzer = ProjectAnalyzer()
        context = AgentContext(project_path=str(sample_project))
        
        result_context = analyzer.execute(context)
        
        assert result_context.project_analysis is not None
        assert "structure" in result_context.project_analysis
        assert "languages" in result_context.project_analysis
        assert result_context.project_analysis.get("llm_enhanced") is False
    
    def test_execute_with_llm(self, sample_project):
        """Test execute with LLM returns enhanced analysis."""
        mock_provider = create_mock_llm_provider()
        analyzer = ProjectAnalyzer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        
        result_context = analyzer.execute(context)
        
        assert result_context.project_analysis is not None
        assert result_context.project_analysis.get("llm_enhanced") is True
        assert "llm_insights" in result_context.project_analysis
        mock_provider.complete.assert_called_once()
    
    def test_llm_usage_tracked(self, sample_project):
        """Test that LLM usage is tracked in context."""
        mock_provider = create_mock_llm_provider()
        analyzer = ProjectAnalyzer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        
        result_context = analyzer.execute(context)
        
        assert result_context.llm_context.total_tokens > 0
        assert result_context.llm_context.estimated_cost > 0
        assert len(result_context.llm_context.responses) == 1


class TestIssueAnalyzerWithLLM:
    """Test cases for IssueAnalyzer with LLM integration."""
    
    def test_init_without_llm(self):
        """Test initialization without LLM provider."""
        analyzer = IssueAnalyzer()
        assert analyzer.llm_provider is None
    
    def test_init_with_llm(self):
        """Test initialization with LLM provider."""
        mock_provider = create_mock_llm_provider()
        analyzer = IssueAnalyzer(llm_provider=mock_provider)
        assert analyzer.llm_provider is mock_provider
    
    def test_execute_without_llm(self, sample_issue, sample_project):
        """Test execute without LLM returns static analysis only."""
        analyzer = IssueAnalyzer()
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        
        result_context = analyzer.execute(context)
        
        assert result_context.issue_analysis is not None
        assert "issue_type" in result_context.issue_analysis
        assert "priority" in result_context.issue_analysis
        assert result_context.issue_analysis.get("llm_enhanced") is False
    
    def test_execute_with_llm(self, sample_issue, sample_project):
        """Test execute with LLM returns enhanced analysis."""
        # Create mock with JSON response
        mock_provider = create_mock_llm_provider()
        mock_provider.complete.return_value = LLMResponse(
            content='```json\n{"issue_type": "bug", "priority": "high", "complexity": "medium"}\n```',
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            finish_reason="stop",
        )
        
        analyzer = IssueAnalyzer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        
        result_context = analyzer.execute(context)
        
        assert result_context.issue_analysis is not None
        assert result_context.issue_analysis.get("llm_enhanced") is True
        assert "llm_insights" in result_context.issue_analysis
        mock_provider.complete.assert_called_once()
    
    def test_parse_json_response(self, sample_issue, sample_project):
        """Test that JSON response from LLM is parsed correctly."""
        mock_provider = create_mock_llm_provider()
        mock_provider.complete.return_value = LLMResponse(
            content='```json\n{"issue_type": "feature", "priority": "medium", "key_requirements": ["req1", "req2"]}\n```',
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            finish_reason="stop",
        )
        
        analyzer = IssueAnalyzer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        
        result_context = analyzer.execute(context)
        
        assert result_context.issue_analysis.get("llm_issue_type") == "feature"
        assert result_context.issue_analysis.get("llm_priority") == "medium"
        assert result_context.issue_analysis.get("llm_requirements") == ["req1", "req2"]


class TestCodeReviewerWithLLM:
    """Test cases for CodeReviewer with LLM integration."""
    
    def test_init_without_llm(self):
        """Test initialization without LLM provider."""
        reviewer = CodeReviewer()
        assert reviewer.llm_provider is None
    
    def test_init_with_llm(self):
        """Test initialization with LLM provider."""
        mock_provider = create_mock_llm_provider()
        reviewer = CodeReviewer(llm_provider=mock_provider)
        assert reviewer.llm_provider is mock_provider
    
    def test_init_with_custom_limits(self):
        """Test initialization with custom file limits."""
        reviewer = CodeReviewer(max_files_for_llm=10, max_lines_per_file=1000)
        assert reviewer.max_files_for_llm == 10
        assert reviewer.max_lines_per_file == 1000
    
    def test_execute_without_llm(self, sample_project, sample_issue):
        """Test execute without LLM returns static analysis only."""
        reviewer = CodeReviewer()
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        context.issue_analysis = {"keywords": ["main"], "affected_files": []}
        
        result_context = reviewer.execute(context)
        
        assert result_context.code_review is not None
        assert "relevant_files" in result_context.code_review
        assert "code_quality" in result_context.code_review
        assert result_context.code_review.get("llm_enhanced") is False
    
    def test_execute_with_llm(self, sample_project, sample_issue):
        """Test execute with LLM returns enhanced review."""
        mock_provider = create_mock_llm_provider()
        mock_provider.complete.return_value = LLMResponse(
            content='```json\n{"quality_score": 8, "issues": [], "strengths": ["clean code"]}\n```',
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            finish_reason="stop",
        )
        
        reviewer = CodeReviewer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        context.issue_analysis = {
            "title": "Test issue",
            "issue_type": "bug",
            "requirements": ["fix bug"],
            "keywords": ["main"],
            "affected_files": ["src/main.py"],
        }
        
        result_context = reviewer.execute(context)
        
        assert result_context.code_review is not None
        assert result_context.code_review.get("llm_enhanced") is True
        assert "llm_insights" in result_context.code_review


class TestAgentLLMFailureHandling:
    """Test cases for agent LLM failure handling."""
    
    def test_project_analyzer_handles_llm_error(self, sample_project):
        """Test that ProjectAnalyzer handles LLM errors gracefully."""
        mock_provider = create_mock_llm_provider()
        mock_provider.complete.side_effect = Exception("LLM API error")
        
        analyzer = ProjectAnalyzer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        
        result_context = analyzer.execute(context)
        
        # Should still complete with static analysis
        assert result_context.project_analysis is not None
        assert result_context.project_analysis.get("llm_enhanced") is False
    
    def test_issue_analyzer_handles_llm_error(self, sample_project, sample_issue):
        """Test that IssueAnalyzer handles LLM errors gracefully."""
        mock_provider = create_mock_llm_provider()
        mock_provider.complete.side_effect = Exception("LLM API error")
        
        analyzer = IssueAnalyzer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        
        result_context = analyzer.execute(context)
        
        # Should still complete with static analysis
        assert result_context.issue_analysis is not None
        assert result_context.issue_analysis.get("llm_enhanced") is False
    
    def test_code_reviewer_handles_llm_error(self, sample_project, sample_issue):
        """Test that CodeReviewer handles LLM errors gracefully."""
        mock_provider = create_mock_llm_provider()
        mock_provider.complete.side_effect = Exception("LLM API error")
        
        reviewer = CodeReviewer(llm_provider=mock_provider)
        context = AgentContext(project_path=str(sample_project))
        context.issue_data = sample_issue
        context.project_analysis = {"languages": {"Python": 5}}
        context.issue_analysis = {"keywords": [], "affected_files": []}
        
        result_context = reviewer.execute(context)
        
        # Should still complete with static analysis
        assert result_context.code_review is not None
        assert result_context.code_review.get("llm_enhanced") is False
