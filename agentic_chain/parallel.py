"""
Parallel Agent Execution - Enables concurrent execution of independent agents.

This module provides infrastructure for running agents in parallel when their
dependencies allow, significantly improving execution performance.

Example execution graph:
    ┌─────────────────┐     ┌──────────────┐
    │ ProjectAnalyzer │     │ IssueAnalyzer│  (Parallel)
    └────────┬────────┘     └──────┬───────┘
             │                      │
             └──────────┬───────────┘
                        ▼
                ┌──────────────┐
                │ CodeReviewer │
                └──────┬───────┘
                       ▼
            ┌──────────────────────┐
            │ SolutionImplementer  │
            └──────────────────────┘
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import AgentContext, BaseAgent

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for the orchestrator."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class AgentStatus(Enum):
    """Status of an agent in the execution graph."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentNode:
    """
    Represents an agent in the dependency graph.
    
    Attributes:
        name: Unique identifier for the agent.
        agent: The agent instance.
        dependencies: Set of agent names this agent depends on.
        status: Current execution status.
        result: Result after execution (the updated context).
        error: Error message if execution failed.
        execution_time: Time taken to execute in seconds.
    """
    name: str
    agent: "BaseAgent"
    dependencies: Set[str] = field(default_factory=set)
    status: AgentStatus = AgentStatus.PENDING
    result: Optional["AgentContext"] = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ParallelExecutionConfig:
    """
    Configuration for parallel execution.
    
    Attributes:
        max_workers: Maximum number of parallel agents (default: 4).
        timeout_seconds: Timeout for each agent execution (default: 300).
        fail_fast: Stop all execution on first failure (default: False).
        continue_on_partial_failure: Continue execution if non-critical agents fail.
    """
    max_workers: int = 4
    timeout_seconds: float = 300.0
    fail_fast: bool = False
    continue_on_partial_failure: bool = True


@dataclass
class ParallelExecutionResult:
    """
    Result of parallel execution.
    
    Attributes:
        success: Whether all agents completed successfully.
        completed_agents: List of successfully completed agent names.
        failed_agents: List of failed agent names with errors.
        skipped_agents: List of skipped agent names.
        total_execution_time: Total wall-clock time for execution.
        agent_times: Individual execution times per agent.
    """
    success: bool
    completed_agents: List[str] = field(default_factory=list)
    failed_agents: List[Dict[str, str]] = field(default_factory=list)
    skipped_agents: List[str] = field(default_factory=list)
    total_execution_time: float = 0.0
    agent_times: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "completed_agents": self.completed_agents,
            "failed_agents": self.failed_agents,
            "skipped_agents": self.skipped_agents,
            "total_execution_time": round(self.total_execution_time, 3),
            "agent_times": {k: round(v, 3) for k, v in self.agent_times.items()},
        }


class DependencyGraph:
    """
    Manages agent dependencies and execution order.
    
    The graph tracks which agents can run in parallel (no dependencies
    on each other) and which must wait for others to complete.
    """
    
    def __init__(self):
        self._nodes: Dict[str, AgentNode] = {}
        self._lock = threading.Lock()
    
    def add_agent(
        self,
        agent: "BaseAgent",
        dependencies: Optional[Set[str]] = None,
    ) -> None:
        """
        Add an agent to the dependency graph.
        
        Args:
            agent: The agent to add.
            dependencies: Set of agent names this agent depends on.
        """
        with self._lock:
            node = AgentNode(
                name=agent.name,
                agent=agent,
                dependencies=dependencies or set(),
            )
            self._nodes[agent.name] = node
    
    def get_node(self, name: str) -> Optional[AgentNode]:
        """Get a node by name."""
        return self._nodes.get(name)
    
    def get_ready_agents(self) -> List[AgentNode]:
        """
        Get agents that are ready to execute.
        
        An agent is ready if:
        - Its status is PENDING
        - All its dependencies have COMPLETED status
        
        Returns:
            List of agent nodes ready for execution.
        """
        with self._lock:
            ready = []
            for node in self._nodes.values():
                if node.status != AgentStatus.PENDING:
                    continue
                
                # Check if all dependencies are completed
                deps_completed = True
                for dep in node.dependencies:
                    dep_node = self._nodes.get(dep)
                    if dep_node is None or dep_node.status != AgentStatus.COMPLETED:
                        deps_completed = False
                        break
                
                if deps_completed:
                    ready.append(node)
            
            return ready
    
    def mark_running(self, name: str) -> None:
        """Mark an agent as running."""
        with self._lock:
            if name in self._nodes:
                self._nodes[name].status = AgentStatus.RUNNING
    
    def mark_completed(
        self,
        name: str,
        result: "AgentContext",
        execution_time: float,
    ) -> None:
        """Mark an agent as completed successfully."""
        with self._lock:
            if name in self._nodes:
                self._nodes[name].status = AgentStatus.COMPLETED
                self._nodes[name].result = result
                self._nodes[name].execution_time = execution_time
    
    def mark_failed(self, name: str, error: str, execution_time: float) -> None:
        """Mark an agent as failed."""
        with self._lock:
            if name in self._nodes:
                self._nodes[name].status = AgentStatus.FAILED
                self._nodes[name].error = error
                self._nodes[name].execution_time = execution_time
    
    def mark_skipped(self, name: str) -> None:
        """Mark an agent as skipped (due to dependency failure)."""
        with self._lock:
            if name in self._nodes:
                self._nodes[name].status = AgentStatus.SKIPPED
    
    def skip_dependents(self, failed_agent: str) -> List[str]:
        """
        Skip all agents that depend on a failed agent (directly or transitively).
        
        Uses a queue-based approach to avoid recursion issues and ensure
        thread-safety.
        
        Returns:
            List of skipped agent names.
        """
        skipped = []
        to_process = [failed_agent]
        processed = set()
        
        with self._lock:
            while to_process:
                current = to_process.pop(0)
                if current in processed:
                    continue
                processed.add(current)
                
                # Find all agents that depend on the current agent
                for node in self._nodes.values():
                    if (node.status == AgentStatus.PENDING and 
                        current in node.dependencies and 
                        node.name not in processed):
                        node.status = AgentStatus.SKIPPED
                        skipped.append(node.name)
                        to_process.append(node.name)
        
        return skipped
    
    def all_completed(self) -> bool:
        """Check if all agents are either completed, failed, or skipped."""
        with self._lock:
            return all(
                node.status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.SKIPPED)
                for node in self._nodes.values()
            )
    
    def has_failures(self) -> bool:
        """Check if any agents failed."""
        with self._lock:
            return any(
                node.status == AgentStatus.FAILED
                for node in self._nodes.values()
            )
    
    def get_results_summary(self) -> Dict[str, List[str]]:
        """Get a summary of execution results."""
        with self._lock:
            return {
                "completed": [n.name for n in self._nodes.values() if n.status == AgentStatus.COMPLETED],
                "failed": [n.name for n in self._nodes.values() if n.status == AgentStatus.FAILED],
                "skipped": [n.name for n in self._nodes.values() if n.status == AgentStatus.SKIPPED],
                "pending": [n.name for n in self._nodes.values() if n.status == AgentStatus.PENDING],
            }
    
    def validate(self) -> List[str]:
        """
        Validate the dependency graph.
        
        Returns:
            List of validation errors, empty if valid.
        """
        errors = []
        
        # Check for missing dependencies
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    errors.append(f"Agent '{node.name}' depends on unknown agent '{dep}'")
        
        # Check for cycles (simple detection)
        visited = set()
        rec_stack = set()
        
        def has_cycle(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)
            
            node = self._nodes.get(name)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.discard(name)
            return False
        
        for node_name in self._nodes:
            if node_name not in visited:
                if has_cycle(node_name):
                    errors.append(f"Circular dependency detected involving '{node_name}'")
        
        return errors


class ThreadSafeContext:
    """
    Thread-safe wrapper for AgentContext to prevent race conditions.
    
    Provides atomic operations for updating context fields.
    """
    
    def __init__(self, context: "AgentContext"):
        self._context = context
        self._lock = threading.RLock()
    
    @property
    def context(self) -> "AgentContext":
        """Get the underlying context (use with caution)."""
        return self._context
    
    def update_project_analysis(self, analysis: dict) -> None:
        """Atomically update project analysis."""
        with self._lock:
            self._context.project_analysis = analysis
    
    def update_issue_analysis(self, analysis: dict) -> None:
        """Atomically update issue analysis."""
        with self._lock:
            self._context.issue_analysis = analysis
    
    def update_code_review(self, review: dict) -> None:
        """Atomically update code review."""
        with self._lock:
            self._context.code_review = review
    
    def update_solution(self, solution: dict) -> None:
        """Atomically update solution."""
        with self._lock:
            self._context.solution = solution
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Atomically update a metadata field."""
        with self._lock:
            self._context.metadata[key] = value
    
    def merge_from(self, other: "AgentContext") -> None:
        """
        Merge results from another context.
        
        Only updates fields that are set in the other context.
        """
        with self._lock:
            if other.project_analysis is not None:
                self._context.project_analysis = other.project_analysis
            if other.issue_analysis is not None:
                self._context.issue_analysis = other.issue_analysis
            if other.code_review is not None:
                self._context.code_review = other.code_review
            if other.solution is not None:
                self._context.solution = other.solution
            
            # Merge metadata
            self._context.metadata.update(other.metadata)
            
            # Merge LLM usage
            if other.llm_context.total_tokens > 0:
                self._context.llm_context.add_usage(
                    prompt_tokens=other.llm_context.total_prompt_tokens,
                    completion_tokens=other.llm_context.total_completion_tokens,
                    cost=other.llm_context.estimated_cost,
                )
                if other.llm_context.provider:
                    self._context.llm_context.provider = other.llm_context.provider
                if other.llm_context.model:
                    self._context.llm_context.model = other.llm_context.model
                self._context.llm_context.responses.extend(other.llm_context.responses)
    
    def create_snapshot(self) -> "AgentContext":
        """Create a snapshot of the current context for an agent to work with."""
        from copy import deepcopy
        from .agents import AgentContext, LLMContext
        
        with self._lock:
            # Create a new context with copied data
            snapshot = AgentContext(project_path=self._context.project_path)
            snapshot.issue_data = deepcopy(self._context.issue_data) if self._context.issue_data else None
            snapshot.project_analysis = deepcopy(self._context.project_analysis) if self._context.project_analysis else None
            snapshot.issue_analysis = deepcopy(self._context.issue_analysis) if self._context.issue_analysis else None
            snapshot.code_review = deepcopy(self._context.code_review) if self._context.code_review else None
            snapshot.solution = deepcopy(self._context.solution) if self._context.solution else None
            snapshot.metadata = deepcopy(self._context.metadata)
            snapshot.llm_context = LLMContext(
                provider=self._context.llm_context.provider,
                model=self._context.llm_context.model,
            )
            return snapshot


class ParallelExecutor:
    """
    Executes agents in parallel based on their dependencies.
    
    Uses a thread pool to run independent agents concurrently while
    respecting dependency constraints.
    """
    
    def __init__(
        self,
        config: Optional[ParallelExecutionConfig] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ):
        """
        Initialize the parallel executor.
        
        Args:
            config: Configuration for parallel execution.
            progress_callback: Optional callback(agent_name, status, progress)
                              for progress tracking.
        """
        self.config = config or ParallelExecutionConfig()
        self.progress_callback = progress_callback
        self._stop_event = threading.Event()
    
    def execute(
        self,
        graph: DependencyGraph,
        context: "AgentContext",
    ) -> ParallelExecutionResult:
        """
        Execute all agents in the graph respecting dependencies.
        
        Args:
            graph: The dependency graph with agents.
            context: The shared agent context.
            
        Returns:
            ParallelExecutionResult with execution details.
        """
        # Validate graph first
        errors = graph.validate()
        if errors:
            raise ValueError(f"Invalid dependency graph: {'; '.join(errors)}")
        
        start_time = time.perf_counter()
        self._stop_event.clear()
        
        # Create thread-safe context wrapper
        safe_context = ThreadSafeContext(context)
        
        # Track agent execution times
        agent_times: Dict[str, float] = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            while not graph.all_completed() and not self._stop_event.is_set():
                # Get agents ready to run
                ready_agents = graph.get_ready_agents()
                
                # Submit ready agents for execution
                for node in ready_agents:
                    if node.name not in futures:
                        graph.mark_running(node.name)
                        self._report_progress(node.name, "running", 0.0)
                        
                        future = executor.submit(
                            self._execute_agent,
                            node,
                            safe_context,
                        )
                        futures[node.name] = future
                
                # Check completed futures
                done_futures = [
                    (name, future) for name, future in futures.items()
                    if future.done()
                ]
                
                for name, future in done_futures:
                    try:
                        exec_time, result_context = future.result(timeout=0)
                        graph.mark_completed(name, result_context, exec_time)
                        agent_times[name] = exec_time
                        safe_context.merge_from(result_context)
                        self._report_progress(name, "completed", 1.0)
                        logger.info(f"Agent '{name}' completed in {exec_time:.3f}s")
                    except Exception as e:
                        error_msg = str(e)
                        graph.mark_failed(name, error_msg, 0.0)
                        self._report_progress(name, "failed", 1.0)
                        logger.error(f"Agent '{name}' failed: {error_msg}")
                        
                        if self.config.fail_fast:
                            self._stop_event.set()
                        else:
                            # Skip dependents of failed agent
                            skipped = graph.skip_dependents(name)
                            for skipped_name in skipped:
                                self._report_progress(skipped_name, "skipped", 1.0)
                    
                    del futures[name]
                
                # Small sleep to prevent busy-waiting
                if not graph.all_completed():
                    time.sleep(0.01)
        
        total_time = time.perf_counter() - start_time
        summary = graph.get_results_summary()
        
        return ParallelExecutionResult(
            success=not graph.has_failures(),
            completed_agents=summary["completed"],
            failed_agents=[
                {"name": n, "error": graph.get_node(n).error or "Unknown error"}
                for n in summary["failed"]
            ],
            skipped_agents=summary["skipped"],
            total_execution_time=total_time,
            agent_times=agent_times,
        )
    
    def _execute_agent(
        self,
        node: AgentNode,
        safe_context: ThreadSafeContext,
    ) -> tuple[float, "AgentContext"]:
        """
        Execute a single agent.
        
        Args:
            node: The agent node to execute.
            safe_context: Thread-safe context wrapper.
            
        Returns:
            Tuple of (execution_time, result_context).
        """
        # Create a snapshot for this agent to work with
        snapshot = safe_context.create_snapshot()
        
        start_time = time.perf_counter()
        
        # Execute the agent
        result_context = node.agent.execute(snapshot)
        
        execution_time = time.perf_counter() - start_time
        
        return execution_time, result_context
    
    def _report_progress(self, agent_name: str, status: str, progress: float) -> None:
        """Report progress if callback is configured."""
        if self.progress_callback:
            try:
                self.progress_callback(agent_name, status, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def stop(self) -> None:
        """Signal the executor to stop."""
        self._stop_event.set()


def create_default_dependency_graph(agents: List["BaseAgent"]) -> DependencyGraph:
    """
    Create a dependency graph with default dependencies for the standard agent pipeline.
    
    The default graph allows ProjectAnalyzer and IssueAnalyzer to run in parallel,
    then CodeReviewer (depends on both), then SolutionImplementer (depends on CodeReviewer).
    
    Args:
        agents: List of agents in the pipeline.
        
    Returns:
        DependencyGraph with standard dependencies configured.
    """
    graph = DependencyGraph()
    
    # Map agent names for lookup
    agent_map = {agent.name: agent for agent in agents}
    
    # Define standard dependencies
    standard_deps = {
        "ProjectAnalyzer": set(),  # No dependencies - can run first
        "IssueAnalyzer": set(),    # No dependencies - can run in parallel with ProjectAnalyzer
        "CodeReviewer": {"ProjectAnalyzer", "IssueAnalyzer"},  # Needs both analyses
        "SolutionImplementer": {"CodeReviewer"},  # Needs code review
    }
    
    # Add agents with their dependencies
    for agent in agents:
        deps = standard_deps.get(agent.name, set())
        # Only include dependencies that actually exist in the agent list
        valid_deps = deps & set(agent_map.keys())
        graph.add_agent(agent, valid_deps)
    
    return graph
