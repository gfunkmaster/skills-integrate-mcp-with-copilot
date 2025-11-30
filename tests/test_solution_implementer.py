"""Tests for LLM integration with SolutionImplementer."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from agentic_chain import AgenticChain, SolutionImplementer, AgentContext
from agentic_chain.agents import LLMContext
from agentic_chain.llm import (
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMUsage,
)


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.config = LLMConfig(provider="mock", model="mock-model", api_key="test")
        self._total_usage = LLMUsage()
    
    @property
    def total_usage(self):
        return self._total_usage
    
    def complete(self, messages, **kwargs):
        usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=0.01)
        self._total_usage = self._total_usage + usage
        return LLMResponse(
            content="Generated code here",
            model="mock-model",
            usage=usage,
            finish_reason="stop",
        )
    
    def generate_code(self, prompt, context=None, language=None, **kwargs):
        return self.complete([])
    
    def generate_implementation_plan(self, issue_description, project_context=None, **kwargs):
        usage = LLMUsage(prompt_tokens=150, completion_tokens=100, total_tokens=250, estimated_cost=0.02)
        self._total_usage = self._total_usage + usage
        return LLMResponse(
            content="# Implementation Plan\n1. Step one\n2. Step two",
            model="mock-model",
            usage=usage,
            finish_reason="stop",
        )


class TestSolutionImplementerWithLLM:
    """Test cases for SolutionImplementer with LLM integration."""
    
    def test_init_without_llm(self):
        """Test initialization without LLM provider."""
        implementer = SolutionImplementer()
        assert implementer.llm_provider is None
    
    def test_init_with_llm(self):
        """Test initialization with LLM provider."""
        mock_llm = MockLLMProvider()
        implementer = SolutionImplementer(llm_provider=mock_llm)
        assert implementer.llm_provider is mock_llm
    
    def test_set_llm_provider(self):
        """Test setting LLM provider after initialization."""
        implementer = SolutionImplementer()
        mock_llm = MockLLMProvider()
        implementer.llm_provider = mock_llm
        assert implementer.llm_provider is mock_llm
    
    def test_execute_without_llm(self, tmp_path):
        """Test execution without LLM uses static analysis."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_analysis = {
            "title": "Test issue",
            "issue_type": "feature",
            "priority": "medium",
            "requirements": ["Add new feature"],
        }
        context.code_review = {"relevant_files": ["app.py"]}
        
        implementer = SolutionImplementer()
        result = implementer.execute(context)
        
        assert result.solution is not None
        assert result.solution.get("llm_generated") is False
        assert "ai_implementation_plan" not in result.solution
    
    def test_execute_with_llm(self, tmp_path):
        """Test execution with LLM generates AI content."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Add feature",
            "body": "We need to add a new feature"
        }
        context.issue_analysis = {
            "title": "Add feature",
            "issue_type": "feature",
            "priority": "medium",
            "requirements": ["Add new feature"],
        }
        context.code_review = {"relevant_files": ["app.py"]}
        context.project_analysis = {"languages": {"Python": 10}}
        
        mock_llm = MockLLMProvider()
        implementer = SolutionImplementer(llm_provider=mock_llm)
        result = implementer.execute(context)
        
        assert result.solution is not None
        assert result.solution.get("llm_generated") is True
        assert "ai_implementation_plan" in result.solution
        assert "code_suggestions" in result.solution
    
    def test_llm_usage_tracking(self, tmp_path):
        """Test that LLM usage is tracked in context."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {"title": "Test", "body": "Test body"}
        context.issue_analysis = {
            "title": "Test",
            "issue_type": "bug",
            "priority": "medium",
            "requirements": ["Fix bug"],
        }
        context.project_analysis = {"languages": {"Python": 5}}
        
        mock_llm = MockLLMProvider()
        implementer = SolutionImplementer(llm_provider=mock_llm)
        result = implementer.execute(context)
        
        # Check usage was tracked
        llm_context = result.llm_context
        assert llm_context.total_tokens > 0
        assert llm_context.estimated_cost > 0
        assert llm_context.provider == "mock"
        assert len(llm_context.responses) > 0


class TestAgenticChainWithLLM:
    """Test cases for AgenticChain with LLM integration."""
    
    def test_init_without_llm(self, tmp_path):
        """Test chain initialization without LLM."""
        chain = AgenticChain(project_path=str(tmp_path))
        assert chain.llm_provider is None
    
    def test_init_with_llm_provider(self, tmp_path):
        """Test chain initialization with LLM provider."""
        mock_llm = MockLLMProvider()
        chain = AgenticChain(project_path=str(tmp_path), llm_provider=mock_llm)
        assert chain.llm_provider is mock_llm
    
    def test_init_with_llm_config(self, tmp_path):
        """Test chain initialization with LLM config dict."""
        with patch('agentic_chain.orchestrator.AgenticChain._create_llm_from_config') as mock_create:
            mock_create.return_value = MockLLMProvider()
            chain = AgenticChain(
                project_path=str(tmp_path),
                llm_config={"provider": "openai", "model": "gpt-4"}
            )
            mock_create.assert_called_once()
    
    def test_set_llm_provider(self, tmp_path):
        """Test setting LLM provider after initialization."""
        chain = AgenticChain(project_path=str(tmp_path))
        mock_llm = MockLLMProvider()
        chain.llm_provider = mock_llm
        
        assert chain.llm_provider is mock_llm
        # Check SolutionImplementer got the provider
        for agent in chain.agents:
            if isinstance(agent, SolutionImplementer):
                assert agent.llm_provider is mock_llm
    
    def test_solve_issue_with_llm(self, tmp_path):
        """Test solving issue with LLM enabled."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        mock_llm = MockLLMProvider()
        chain = AgenticChain(project_path=str(tmp_path), llm_provider=mock_llm)
        
        issue = {
            "title": "Add new feature",
            "body": "We need to add a greeting feature",
            "labels": [{"name": "feature"}]
        }
        
        result = chain.solve_issue(issue)
        
        assert result is not None
        assert result["solution"]["llm_generated"] is True
        assert "ai_implementation_plan" in result["solution"]
    
    def test_get_llm_usage(self, tmp_path):
        """Test getting LLM usage statistics."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        mock_llm = MockLLMProvider()
        chain = AgenticChain(project_path=str(tmp_path), llm_provider=mock_llm)
        
        chain.solve_issue({
            "title": "Test",
            "body": "Test",
            "labels": []
        })
        
        usage = chain.get_llm_usage()
        
        assert usage["total_tokens"] > 0
        assert usage["provider"] == "mock"
    
    def test_solution_summary_with_llm(self, tmp_path):
        """Test solution summary includes LLM info."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        mock_llm = MockLLMProvider()
        chain = AgenticChain(project_path=str(tmp_path), llm_provider=mock_llm)
        
        chain.solve_issue({
            "title": "Test feature",
            "body": "Test body",
            "labels": []
        })
        
        summary = chain.get_solution_summary()
        
        assert "AI-generated solution" in summary
        assert "LLM USAGE" in summary
    
    def test_repr_with_llm(self, tmp_path):
        """Test string representation includes LLM info."""
        mock_llm = MockLLMProvider()
        chain = AgenticChain(project_path=str(tmp_path), llm_provider=mock_llm)
        
        repr_str = repr(chain)
        
        assert "llm=mock" in repr_str


class TestLLMContext:
    """Test cases for LLMContext."""
    
    def test_default_values(self):
        """Test default LLMContext values."""
        ctx = LLMContext()
        assert ctx.provider is None
        assert ctx.total_tokens == 0
        assert ctx.estimated_cost == 0.0
        assert ctx.responses == []
    
    def test_add_usage(self):
        """Test adding usage to context."""
        ctx = LLMContext()
        ctx.add_usage(prompt_tokens=100, completion_tokens=50, cost=0.01)
        
        assert ctx.total_prompt_tokens == 100
        assert ctx.total_completion_tokens == 50
        assert ctx.total_tokens == 150
        assert ctx.estimated_cost == 0.01
    
    def test_multiple_add_usage(self):
        """Test multiple usage additions."""
        ctx = LLMContext()
        ctx.add_usage(100, 50, 0.01)
        ctx.add_usage(200, 100, 0.02)
        
        assert ctx.total_prompt_tokens == 300
        assert ctx.total_completion_tokens == 150
        assert ctx.total_tokens == 450
        assert ctx.estimated_cost == 0.03
    
    def test_to_dict(self):
        """Test serialization to dict."""
        ctx = LLMContext(provider="openai", model="gpt-4")
        ctx.add_usage(100, 50, 0.01)
        ctx.responses.append({"type": "test"})
        
        result = ctx.to_dict()
        
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4"
        assert result["total_tokens"] == 150
        assert result["response_count"] == 1
