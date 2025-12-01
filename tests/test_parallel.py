"""Tests for parallel agent execution."""

import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_chain import AgenticChain
from agentic_chain.agents import AgentContext, BaseAgent
from agentic_chain.parallel import (
    ExecutionMode,
    ParallelExecutionConfig,
    ParallelExecutionResult,
    ParallelExecutor,
    DependencyGraph,
    AgentNode,
    AgentStatus,
    ThreadSafeContext,
    create_default_dependency_graph,
)


class SlowMockAgent(BaseAgent):
    """Mock agent that simulates work by sleeping."""
    
    def __init__(self, name: str, sleep_time: float = 0.1, should_fail: bool = False):
        super().__init__(name)
        self.sleep_time = sleep_time
        self.should_fail = should_fail
        self.executed = False
        self.execution_count = 0
        
    def execute(self, context: AgentContext) -> AgentContext:
        time.sleep(self.sleep_time)
        self.executed = True
        self.execution_count += 1
        
        if self.should_fail:
            raise RuntimeError(f"Agent {self.name} failed intentionally")
        
        # Store results in metadata based on agent type
        if self.name == "ProjectAnalyzer":
            context.project_analysis = {
                "structure": {"directories": [], "file_count": 0},
                "languages": {"Python": 1},
            }
        elif self.name == "IssueAnalyzer":
            context.issue_analysis = {
                "title": context.issue_data.get("title", ""),
                "issue_type": "feature",
            }
        elif self.name == "CodeReviewer":
            context.code_review = {
                "relevant_files": [],
                "code_quality": {"has_tests": True},
            }
        elif self.name == "SolutionImplementer":
            context.solution = {
                "proposed_changes": [],
                "implementation_plan": {},
            }
        
        context.metadata[f"{self.name}_executed"] = True
        return context


class TestDependencyGraph:
    """Tests for DependencyGraph class."""
    
    def test_add_agent(self):
        """Test adding an agent to the graph."""
        graph = DependencyGraph()
        agent = SlowMockAgent("TestAgent")
        
        graph.add_agent(agent, dependencies=set())
        
        node = graph.get_node("TestAgent")
        assert node is not None
        assert node.name == "TestAgent"
        assert node.status == AgentStatus.PENDING
    
    def test_add_agent_with_dependencies(self):
        """Test adding an agent with dependencies."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1")
        agent2 = SlowMockAgent("Agent2")
        
        graph.add_agent(agent1)
        graph.add_agent(agent2, dependencies={"Agent1"})
        
        node = graph.get_node("Agent2")
        assert "Agent1" in node.dependencies
    
    def test_get_ready_agents_no_dependencies(self):
        """Test getting ready agents when none have dependencies."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1")
        agent2 = SlowMockAgent("Agent2")
        
        graph.add_agent(agent1)
        graph.add_agent(agent2)
        
        ready = graph.get_ready_agents()
        assert len(ready) == 2
    
    def test_get_ready_agents_with_dependencies(self):
        """Test getting ready agents respects dependencies."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1")
        agent2 = SlowMockAgent("Agent2")
        
        graph.add_agent(agent1)
        graph.add_agent(agent2, dependencies={"Agent1"})
        
        ready = graph.get_ready_agents()
        assert len(ready) == 1
        assert ready[0].name == "Agent1"
    
    def test_mark_completed(self):
        """Test marking an agent as completed."""
        graph = DependencyGraph()
        agent = SlowMockAgent("Agent1")
        graph.add_agent(agent)
        
        context = AgentContext(project_path="/tmp")
        graph.mark_completed("Agent1", context, 1.5)
        
        node = graph.get_node("Agent1")
        assert node.status == AgentStatus.COMPLETED
        assert node.execution_time == 1.5
    
    def test_mark_failed(self):
        """Test marking an agent as failed."""
        graph = DependencyGraph()
        agent = SlowMockAgent("Agent1")
        graph.add_agent(agent)
        
        graph.mark_failed("Agent1", "Test error", 0.5)
        
        node = graph.get_node("Agent1")
        assert node.status == AgentStatus.FAILED
        assert node.error == "Test error"
    
    def test_skip_dependents(self):
        """Test skipping agents that depend on a failed agent."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1")
        agent2 = SlowMockAgent("Agent2")
        agent3 = SlowMockAgent("Agent3")
        
        graph.add_agent(agent1)
        graph.add_agent(agent2, dependencies={"Agent1"})
        graph.add_agent(agent3, dependencies={"Agent2"})
        
        graph.mark_failed("Agent1", "Failed", 0.1)
        skipped = graph.skip_dependents("Agent1")
        
        assert "Agent2" in skipped
        assert "Agent3" in skipped
    
    def test_validate_missing_dependency(self):
        """Test validation catches missing dependencies."""
        graph = DependencyGraph()
        agent = SlowMockAgent("Agent1")
        
        graph.add_agent(agent, dependencies={"NonExistent"})
        
        errors = graph.validate()
        assert len(errors) > 0
        assert "NonExistent" in errors[0]
    
    def test_validate_no_errors(self):
        """Test validation passes for valid graph."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1")
        agent2 = SlowMockAgent("Agent2")
        
        graph.add_agent(agent1)
        graph.add_agent(agent2, dependencies={"Agent1"})
        
        errors = graph.validate()
        assert len(errors) == 0


class TestThreadSafeContext:
    """Tests for ThreadSafeContext class."""
    
    def test_update_project_analysis(self):
        """Test atomic update of project analysis."""
        context = AgentContext(project_path="/tmp")
        safe_context = ThreadSafeContext(context)
        
        safe_context.update_project_analysis({"test": "value"})
        
        assert context.project_analysis == {"test": "value"}
    
    def test_update_issue_analysis(self):
        """Test atomic update of issue analysis."""
        context = AgentContext(project_path="/tmp")
        safe_context = ThreadSafeContext(context)
        
        safe_context.update_issue_analysis({"test": "value"})
        
        assert context.issue_analysis == {"test": "value"}
    
    def test_merge_from(self):
        """Test merging results from another context."""
        context1 = AgentContext(project_path="/tmp")
        context2 = AgentContext(project_path="/tmp")
        context2.project_analysis = {"merged": True}
        
        safe_context = ThreadSafeContext(context1)
        safe_context.merge_from(context2)
        
        assert context1.project_analysis == {"merged": True}
    
    def test_create_snapshot(self):
        """Test creating a snapshot of context."""
        context = AgentContext(project_path="/tmp")
        context.project_analysis = {"original": True}
        
        safe_context = ThreadSafeContext(context)
        snapshot = safe_context.create_snapshot()
        
        # Snapshot should have the data
        assert snapshot.project_analysis == {"original": True}
        
        # Modifying snapshot should not affect original
        snapshot.project_analysis["modified"] = True
        assert "modified" not in context.project_analysis


class TestParallelExecutor:
    """Tests for ParallelExecutor class."""
    
    def test_execute_simple_graph(self, tmp_path):
        """Test executing a simple graph with no dependencies."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1", sleep_time=0.05)
        agent2 = SlowMockAgent("Agent2", sleep_time=0.05)
        
        graph.add_agent(agent1)
        graph.add_agent(agent2)
        
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {"title": "Test"}
        
        executor = ParallelExecutor(config=ParallelExecutionConfig(max_workers=2))
        result = executor.execute(graph, context)
        
        assert result.success
        assert len(result.completed_agents) == 2
        assert agent1.executed
        assert agent2.executed
    
    def test_execute_with_dependencies(self, tmp_path):
        """Test executing graph with dependencies."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("ProjectAnalyzer", sleep_time=0.05)
        agent2 = SlowMockAgent("IssueAnalyzer", sleep_time=0.05)
        agent3 = SlowMockAgent("CodeReviewer", sleep_time=0.05)
        
        graph.add_agent(agent1)
        graph.add_agent(agent2)
        graph.add_agent(agent3, dependencies={"ProjectAnalyzer", "IssueAnalyzer"})
        
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {"title": "Test"}
        
        executor = ParallelExecutor(config=ParallelExecutionConfig(max_workers=2))
        result = executor.execute(graph, context)
        
        assert result.success
        assert len(result.completed_agents) == 3
    
    def test_execute_handles_failure(self, tmp_path):
        """Test execution handles agent failure."""
        graph = DependencyGraph()
        agent1 = SlowMockAgent("Agent1", sleep_time=0.05, should_fail=True)
        agent2 = SlowMockAgent("Agent2", sleep_time=0.05)
        
        graph.add_agent(agent1)
        graph.add_agent(agent2, dependencies={"Agent1"})
        
        context = AgentContext(project_path=str(tmp_path))
        
        config = ParallelExecutionConfig(max_workers=2, continue_on_partial_failure=True)
        executor = ParallelExecutor(config=config)
        result = executor.execute(graph, context)
        
        assert not result.success
        assert len(result.failed_agents) == 1
        assert len(result.skipped_agents) == 1
    
    def test_execute_parallel_is_faster(self, tmp_path):
        """Test that parallel execution is faster than sequential for independent agents."""
        # Create 4 independent agents, each taking 0.1 seconds
        sleep_time = 0.1
        agents = [SlowMockAgent(f"Agent{i}", sleep_time=sleep_time) for i in range(4)]
        
        # Parallel execution with 4 workers
        graph = DependencyGraph()
        for agent in agents:
            graph.add_agent(agent)
        
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {"title": "Test"}
        
        executor = ParallelExecutor(config=ParallelExecutionConfig(max_workers=4))
        result = executor.execute(graph, context)
        
        # With 4 parallel workers, all agents should complete in ~0.1s, not ~0.4s
        # Allow some overhead margin
        assert result.total_execution_time < sleep_time * 3
    
    def test_progress_callback(self, tmp_path):
        """Test progress callback is called."""
        progress_calls = []
        
        def callback(agent_name, status, progress):
            progress_calls.append((agent_name, status, progress))
        
        graph = DependencyGraph()
        agent = SlowMockAgent("TestAgent", sleep_time=0.01)
        graph.add_agent(agent)
        
        context = AgentContext(project_path=str(tmp_path))
        
        executor = ParallelExecutor(
            config=ParallelExecutionConfig(max_workers=1),
            progress_callback=callback,
        )
        executor.execute(graph, context)
        
        assert len(progress_calls) > 0


class TestCreateDefaultDependencyGraph:
    """Tests for create_default_dependency_graph function."""
    
    def test_creates_correct_dependencies(self):
        """Test that default graph has correct dependencies."""
        agents = [
            SlowMockAgent("ProjectAnalyzer"),
            SlowMockAgent("IssueAnalyzer"),
            SlowMockAgent("CodeReviewer"),
            SlowMockAgent("SolutionImplementer"),
        ]
        
        graph = create_default_dependency_graph(agents)
        
        # ProjectAnalyzer should have no dependencies
        pa_node = graph.get_node("ProjectAnalyzer")
        assert len(pa_node.dependencies) == 0
        
        # IssueAnalyzer should have no dependencies
        ia_node = graph.get_node("IssueAnalyzer")
        assert len(ia_node.dependencies) == 0
        
        # CodeReviewer should depend on both
        cr_node = graph.get_node("CodeReviewer")
        assert "ProjectAnalyzer" in cr_node.dependencies
        assert "IssueAnalyzer" in cr_node.dependencies
        
        # SolutionImplementer should depend on CodeReviewer
        si_node = graph.get_node("SolutionImplementer")
        assert "CodeReviewer" in si_node.dependencies
    
    def test_handles_missing_agents(self):
        """Test graph handles missing standard agents gracefully."""
        agents = [
            SlowMockAgent("ProjectAnalyzer"),
            SlowMockAgent("CustomAgent"),
        ]
        
        graph = create_default_dependency_graph(agents)
        
        # Should still work even with non-standard agents
        assert graph.get_node("ProjectAnalyzer") is not None
        assert graph.get_node("CustomAgent") is not None


class TestAgenticChainParallel:
    """Tests for parallel execution in AgenticChain."""
    
    def test_init_with_parallel_mode(self, tmp_path):
        """Test chain initialization with parallel mode."""
        chain = AgenticChain(
            project_path=str(tmp_path),
            execution_mode=ExecutionMode.PARALLEL,
        )
        
        assert chain.execution_mode == ExecutionMode.PARALLEL
    
    def test_init_with_parallel_config(self, tmp_path):
        """Test chain initialization with custom parallel config."""
        config = ParallelExecutionConfig(max_workers=8, timeout_seconds=60)
        chain = AgenticChain(
            project_path=str(tmp_path),
            execution_mode=ExecutionMode.PARALLEL,
            parallel_config=config,
        )
        
        assert chain.parallel_config.max_workers == 8
        assert chain.parallel_config.timeout_seconds == 60
    
    def test_solve_issue_sequential_mode(self, tmp_path):
        """Test solve_issue in sequential mode (default)."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path))
        result = chain.solve_issue({
            "title": "Test Issue",
            "body": "Test description",
            "labels": [],
        })
        
        assert result is not None
        assert "project_analysis" in result
        assert "issue_analysis" in result
    
    def test_solve_issue_parallel_mode(self, tmp_path):
        """Test solve_issue in parallel mode."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(
            project_path=str(tmp_path),
            execution_mode=ExecutionMode.PARALLEL,
            parallel_config=ParallelExecutionConfig(max_workers=4),
        )
        
        result = chain.solve_issue({
            "title": "Test Issue",
            "body": "Test description",
            "labels": [],
        })
        
        assert result is not None
        assert "project_analysis" in result
        assert "issue_analysis" in result
        assert "code_review" in result
        assert "solution" in result
    
    def test_parallel_execution_result_available(self, tmp_path):
        """Test that parallel execution result is available after solve_issue."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(
            project_path=str(tmp_path),
            execution_mode=ExecutionMode.PARALLEL,
        )
        
        chain.solve_issue({
            "title": "Test Issue",
            "body": "Test description",
            "labels": [],
        })
        
        result = chain.last_parallel_result
        assert result is not None
        assert isinstance(result, ParallelExecutionResult)
        assert len(result.completed_agents) == 4
    
    def test_progress_callback_in_chain(self, tmp_path):
        """Test progress callback works through chain."""
        progress_calls = []
        
        def callback(agent_name, status, progress):
            progress_calls.append((agent_name, status, progress))
        
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(
            project_path=str(tmp_path),
            execution_mode=ExecutionMode.PARALLEL,
            progress_callback=callback,
        )
        
        chain.solve_issue({
            "title": "Test Issue",
            "body": "Test description",
            "labels": [],
        })
        
        assert len(progress_calls) > 0
    
    def test_mode_switching(self, tmp_path):
        """Test switching between sequential and parallel modes."""
        chain = AgenticChain(project_path=str(tmp_path))
        
        assert chain.execution_mode == ExecutionMode.SEQUENTIAL
        
        chain.execution_mode = ExecutionMode.PARALLEL
        assert chain.execution_mode == ExecutionMode.PARALLEL
        
        chain.execution_mode = ExecutionMode.SEQUENTIAL
        assert chain.execution_mode == ExecutionMode.SEQUENTIAL


class TestParallelExecutionResult:
    """Tests for ParallelExecutionResult class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ParallelExecutionResult(
            success=True,
            completed_agents=["Agent1", "Agent2"],
            failed_agents=[{"name": "Agent3", "error": "Test error"}],
            skipped_agents=["Agent4"],
            total_execution_time=1.234,
            agent_times={"Agent1": 0.5, "Agent2": 0.7},
        )
        
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["completed_agents"] == ["Agent1", "Agent2"]
        assert len(data["failed_agents"]) == 1
        assert data["total_execution_time"] == 1.234
