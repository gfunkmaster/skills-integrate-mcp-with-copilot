"""Tests for the AgenticChain orchestrator."""

import json
import pytest
from pathlib import Path

from agentic_chain import AgenticChain
from agentic_chain.agents import BaseAgent, AgentContext


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str = "MockAgent"):
        super().__init__(name)
        self.executed = False
        
    def execute(self, context: AgentContext) -> AgentContext:
        self.executed = True
        context.metadata["mock_agent"] = True
        return context


class TestAgenticChain:
    """Test cases for AgenticChain orchestrator."""
    
    def test_init(self, tmp_path):
        """Test chain initialization."""
        chain = AgenticChain(project_path=str(tmp_path))
        
        assert chain.project_path == tmp_path.resolve()
        assert len(chain.agents) == 4  # Default agents
        
    def test_init_with_custom_agents(self, tmp_path):
        """Test chain initialization with custom agents."""
        custom_agent = MockAgent("CustomAgent")
        chain = AgenticChain(
            project_path=str(tmp_path),
            custom_agents=[custom_agent]
        )
        
        assert len(chain.agents) == 5
        
    def test_add_agent(self, tmp_path):
        """Test adding an agent to the chain."""
        chain = AgenticChain(project_path=str(tmp_path))
        initial_count = len(chain.agents)
        
        chain.add_agent(MockAgent("TestAgent"))
        
        assert len(chain.agents) == initial_count + 1
        
    def test_add_agent_at_position(self, tmp_path):
        """Test adding an agent at a specific position."""
        chain = AgenticChain(project_path=str(tmp_path))
        
        chain.add_agent(MockAgent("FirstAgent"), position=0)
        
        assert chain.agents[0].name == "FirstAgent"
        
    def test_remove_agent(self, tmp_path):
        """Test removing an agent from the chain."""
        chain = AgenticChain(project_path=str(tmp_path))
        initial_count = len(chain.agents)
        
        result = chain.remove_agent("ProjectAnalyzer")
        
        assert result is True
        assert len(chain.agents) == initial_count - 1
        
    def test_remove_nonexistent_agent(self, tmp_path):
        """Test removing a nonexistent agent."""
        chain = AgenticChain(project_path=str(tmp_path))
        
        result = chain.remove_agent("NonexistentAgent")
        
        assert result is False
        
    def test_analyze_project(self, tmp_path):
        """Test project analysis."""
        # Create a minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Test")
        
        chain = AgenticChain(project_path=str(tmp_path))
        result = chain.analyze_project()
        
        assert result is not None
        assert "structure" in result
        assert "languages" in result
        
    def test_solve_issue(self, tmp_path):
        """Test solving an issue."""
        # Create a minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path))
        issue = {
            "title": "Add feature",
            "body": "We need to add a new feature",
            "labels": [{"name": "feature"}]
        }
        
        result = chain.solve_issue(issue)
        
        assert result is not None
        assert "project_analysis" in result
        assert "issue_analysis" in result
        assert "code_review" in result
        assert "solution" in result
        
    def test_get_result(self, tmp_path):
        """Test getting results."""
        chain = AgenticChain(project_path=str(tmp_path))
        result = chain.get_result()
        
        assert "project_path" in result
        assert result["project_path"] == str(tmp_path.resolve())
        
    def test_get_solution_summary_before_execution(self, tmp_path):
        """Test getting summary before execution."""
        chain = AgenticChain(project_path=str(tmp_path))
        summary = chain.get_solution_summary()
        
        assert "has not been executed" in summary
        
    def test_get_solution_summary_after_execution(self, tmp_path):
        """Test getting summary after execution."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path))
        chain.solve_issue({
            "title": "Test issue",
            "body": "Test body",
            "labels": []
        })
        
        summary = chain.get_solution_summary()
        
        assert "AGENTIC CHAIN SOLUTION SUMMARY" in summary
        assert "ISSUE ANALYSIS" in summary
        assert "PROPOSED SOLUTION" in summary
        
    def test_export_result(self, tmp_path):
        """Test exporting results to JSON."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path))
        chain.solve_issue({
            "title": "Test",
            "body": "Test",
            "labels": []
        })
        
        output_path = tmp_path / "result.json"
        chain.export_result(str(output_path))
        
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "project_analysis" in data
        assert "solution" in data
        
    def test_repr(self, tmp_path):
        """Test string representation."""
        chain = AgenticChain(project_path=str(tmp_path))
        repr_str = repr(chain)
        
        assert "AgenticChain" in repr_str
        assert "ProjectAnalyzer" in repr_str
