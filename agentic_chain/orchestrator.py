"""
Orchestrator - Chains agents together to solve issues.

Supports both sequential and parallel execution modes for optimal performance.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .agents import AgentContext, BaseAgent
from .agents.project_analyzer import ProjectAnalyzer
from .agents.issue_analyzer import IssueAnalyzer
from .agents.code_reviewer import CodeReviewer
from .agents.solution_implementer import SolutionImplementer
from .interactive import (
    InteractionHandler,
    ConsoleInteractionHandler,
    InteractionType,
    InteractionPoint,
    InteractionOption,
    InteractionResult,
    InteractionHistory,
)
from .observability import (
    Tracer,
    TracerConfig,
    SpanKind,
    SpanStatus,
    MetricsCollector,
    StructuredLogger,
    LogLevel,
    TraceContext,
    ContextManager,
    ExecutionTimeline,
    AgentStep,
    ObservabilityData,
)
from .parallel import (
    ExecutionMode,
    ParallelExecutionConfig,
    ParallelExecutionResult,
    ParallelExecutor,
    DependencyGraph,
    create_default_dependency_graph,
)


logger = logging.getLogger(__name__)


class AgenticChain:
    """
    Orchestrates a chain of agents to analyze projects, understand issues,
    review code, and propose solutions.
    
    Supports LLM integration for intelligent code generation.
    Includes comprehensive observability with tracing, metrics, and logging.
    Supports interactive mode for human-in-the-loop processing.
    Supports both sequential and parallel execution modes for optimal performance.
    
    Example usage:
        # Basic usage (static analysis)
        chain = AgenticChain(project_path="/path/to/project")
        result = chain.solve_issue({
            "title": "Bug in login",
            "body": "Users cannot login when..."
        })
        
        # With LLM integration
        from agentic_chain import LLMFactory
        llm = LLMFactory.create("openai", model="gpt-4")
        chain = AgenticChain(project_path="/path/to/project", llm_provider=llm)
        result = chain.solve_issue(issue_data)
        
        # With observability
        chain = AgenticChain(project_path="/path/to/project", enable_tracing=True)
        result = chain.solve_issue(issue_data)
        timeline = chain.get_execution_timeline()
        
        # With interactive mode
        chain = AgenticChain(project_path="/path/to/project", interactive=True)
        result = chain.solve_issue(issue_data)
        history = chain.get_interaction_history()
        # With parallel execution
        chain = AgenticChain(
            project_path="/path/to/project",
            execution_mode=ExecutionMode.PARALLEL,
            parallel_config=ParallelExecutionConfig(max_workers=4),
        )
        result = chain.solve_issue(issue_data)
    """
    
    def __init__(
        self,
        project_path: str,
        custom_agents: Optional[list] = None,
        llm_provider: Optional["LLMProvider"] = None,
        llm_config: Optional[dict] = None,
        enable_tracing: bool = True,
        tracer_config: Optional[TracerConfig] = None,
        interactive: bool = False,
        interaction_handler: Optional[InteractionHandler] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        parallel_config: Optional[ParallelExecutionConfig] = None,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ):
        """
        Initialize the agentic chain.
        
        Args:
            project_path: Path to the project to analyze.
            custom_agents: Optional list of custom agents to include.
            llm_provider: Optional LLM provider for AI-powered generation.
            llm_config: Optional dict to create LLM provider. Keys:
                - provider: "openai" or "anthropic"
                - model: Model name (optional)
                - api_key: API key (optional, uses env var if not set)
            enable_tracing: Whether to enable distributed tracing.
            tracer_config: Optional configuration for the tracer.
            interactive: Whether to enable interactive mode.
            interaction_handler: Custom interaction handler (defaults to ConsoleInteractionHandler).
            execution_mode: Sequential or parallel execution mode.
            parallel_config: Configuration for parallel execution.
            progress_callback: Optional callback(agent_name, status, progress)
                              for progress tracking during parallel execution.
        """
        self.project_path = Path(project_path).resolve()
        
        # Set up interactive mode (lazy initialization - only create handler when needed)
        self._interactive = interactive
        if interaction_handler:
            self._interaction_handler = interaction_handler
        elif interactive:
            self._interaction_handler = ConsoleInteractionHandler(enabled=True)
        else:
            self._interaction_handler = None  # No handler created until needed
        
        self.context = AgentContext(
            project_path=str(self.project_path),
            interaction_handler=self._interaction_handler,
        )
        
        # Set up observability
        self._tracer = Tracer(tracer_config or TracerConfig(enabled=enable_tracing))
        self._metrics = MetricsCollector()
        self._current_trace_id: Optional[str] = None
        self._execution_timelines: List[ExecutionTimeline] = []
        
        # Parallel execution configuration
        self._execution_mode = execution_mode
        self._parallel_config = parallel_config or ParallelExecutionConfig()
        self._progress_callback = progress_callback
        self._last_parallel_result: Optional[ParallelExecutionResult] = None
        
        # Set up LLM provider
        self._llm_provider = llm_provider
        if not self._llm_provider and llm_config:
            self._llm_provider = self._create_llm_from_config(llm_config)
        
        # Create agents with LLM if available
        project_analyzer = ProjectAnalyzer()
        issue_analyzer = IssueAnalyzer()
        code_reviewer = CodeReviewer()
        solution_implementer = SolutionImplementer()
        
        if self._llm_provider:
            project_analyzer.llm_provider = self._llm_provider
            issue_analyzer.llm_provider = self._llm_provider
            code_reviewer.llm_provider = self._llm_provider
            solution_implementer.llm_provider = self._llm_provider
        
        # Default agent pipeline
        self.agents = [
            project_analyzer,
            issue_analyzer,
            code_reviewer,
            solution_implementer,
        ]
        
        # Add custom agents if provided
        if custom_agents:
            for agent in custom_agents:
                if isinstance(agent, BaseAgent):
                    self.agents.append(agent)
                    
        self._executed = False
    
    def _create_llm_from_config(self, config: dict) -> Optional["LLMProvider"]:
        """Create LLM provider from config dict."""
        try:
            from .llm import LLMFactory
            return LLMFactory.create(**config)
        except Exception as e:
            logger.warning(f"Failed to create LLM provider: {e}")
            return None
    
    @property
    def llm_provider(self) -> Optional["LLMProvider"]:
        """Get the LLM provider."""
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider: "LLMProvider"):
        """Set the LLM provider and update agents."""
        self._llm_provider = provider
        # Update all LLM-capable agents with new provider
        for agent in self.agents:
            if hasattr(agent, 'llm_provider'):
                agent.llm_provider = provider
    
    @property
    def tracer(self) -> Tracer:
        """Get the tracer instance."""
        return self._tracer
    
    @property
    def metrics(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._metrics
    
    @property
    def interactive(self) -> bool:
        """Check if interactive mode is enabled."""
        return self._interactive
    
    @interactive.setter
    def interactive(self, value: bool):
        """Enable or disable interactive mode."""
        self._interactive = value
        if value and self._interaction_handler is None:
            # Lazy initialization when enabling interactive mode
            self._interaction_handler = ConsoleInteractionHandler(enabled=True)
            self.context.interaction_handler = self._interaction_handler
        elif self._interaction_handler:
            self._interaction_handler.enabled = value
    
    @property
    def interaction_handler(self) -> Optional[InteractionHandler]:
        """Get the interaction handler."""
        return self._interaction_handler
    
    def get_interaction_history(self) -> Optional[InteractionHistory]:
        """
        Get the interaction history from the current session.
        
        Returns:
            InteractionHistory if available, None otherwise.
        """
        if self._interaction_handler:
            return self._interaction_handler.history
        return None
    
    def _handle_agent_interaction(self, agent: BaseAgent) -> bool:
        """
        Handle interactive review after an agent completes.
        
        Returns True to continue, False to cancel.
        """
        # Define interaction points based on agent type
        if agent.name == "IssueAnalyzer":
            return self._review_issue_analysis()
        elif agent.name == "CodeReviewer":
            return self._review_code_analysis()
        
        return True  # Continue by default
    
    def _review_issue_analysis(self) -> bool:
        """Interactive review of issue analysis results."""
        if not self.context.issue_analysis:
            return True
        
        analysis = self.context.issue_analysis
        issue_type = analysis.get("issue_type", "unknown")
        priority = analysis.get("priority", "medium")
        requirements = analysis.get("requirements", [])
        
        message = f"Issue Type: {issue_type}\nPriority: {priority}"
        if requirements:
            message += "\n\nRequirements identified:"
            for i, req in enumerate(requirements[:5], 1):
                message += f"\n  {i}. {req[:80]}"
        
        result = self._interaction_handler.request_confirmation(
            title="Issue Analysis Complete",
            message=message,
            agent_name="IssueAnalyzer",
            default=True,
        )
        
        # Handle user feedback
        if result.custom_input or result.feedback:
            self.context.metadata["user_feedback_analysis"] = (
                result.custom_input or result.feedback
            )
        
        return result.approved
    
    def _review_code_analysis(self) -> bool:
        """Interactive review of code review results."""
        if not self.context.code_review:
            return True
        
        review = self.context.code_review
        relevant_files = review.get("relevant_files", [])
        issues_found = review.get("potential_issues", [])
        
        message = f"Found {len(relevant_files)} relevant files"
        if relevant_files:
            message += ":\n"
            for f in relevant_files[:5]:
                message += f"  • {f}\n"
        
        if issues_found:
            message += f"\n{len(issues_found)} potential issues identified"
        
        result = self._interaction_handler.request_confirmation(
            title="Code Review Complete",
            message=message,
            agent_name="CodeReviewer",
            default=True,
        )
        
        return result.approved
    
    def _handle_solution_review(self) -> bool:
        """Interactive review of the final proposed solution."""
        if not self.context.solution:
            return True
        
        solution = self.context.solution
        proposed_changes = solution.get("proposed_changes", [])
        risks = solution.get("risks", [])
        implementation_plan = solution.get("implementation_plan", {})
        
        summary = "Proposed Solution:\n"
        
        if implementation_plan:
            complexity = implementation_plan.get("complexity", "unknown")
            hours = implementation_plan.get("estimated_hours", "N/A")
            summary += f"\nComplexity: {complexity}, Estimated: {hours} hours"
        
        result = self._interaction_handler.request_solution_review(
            title="Solution Review",
            solution_summary=summary,
            proposed_changes=proposed_changes,
            risks=risks,
            agent_name="SolutionImplementer",
        )
        
        # Store user feedback for potential refinement
        if result.custom_input or result.feedback:
            self.context.metadata["user_feedback_solution"] = (
                result.custom_input or result.feedback
            )
            self.context.metadata["solution_modified"] = True
        
        return result.approved

    @property
    def execution_mode(self) -> ExecutionMode:
        """Get the current execution mode."""
        return self._execution_mode
    
    @execution_mode.setter
    def execution_mode(self, mode: ExecutionMode):
        """Set the execution mode."""
        self._execution_mode = mode
    
    @property
    def parallel_config(self) -> ParallelExecutionConfig:
        """Get the parallel execution configuration."""
        return self._parallel_config
    
    @parallel_config.setter
    def parallel_config(self, config: ParallelExecutionConfig):
        """Set the parallel execution configuration."""
        self._parallel_config = config
    
    @property
    def last_parallel_result(self) -> Optional[ParallelExecutionResult]:
        """Get the result of the last parallel execution."""
        return self._last_parallel_result
        
    def add_agent(self, agent: BaseAgent, position: Optional[int] = None):
        """
        Add an agent to the chain.
        
        Args:
            agent: The agent to add.
            position: Optional position in the chain. If None, appends to end.
        """
        if position is not None:
            self.agents.insert(position, agent)
        else:
            self.agents.append(agent)
            
    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the chain by name.
        
        Args:
            agent_name: Name of the agent to remove.
            
        Returns:
            True if agent was removed, False if not found.
        """
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                self.agents.pop(i)
                return True
        return False
        
    def solve_issue(self, issue_data: dict) -> dict:
        """
        Run the full agentic chain to solve an issue.
        
        Supports both sequential and parallel execution modes based on configuration.
        All agent executions are traced and metrics are collected.
        
        Args:
            issue_data: Dictionary containing issue information:
                - title: Issue title
                - body: Issue description
                - labels: Optional list of labels
                - number: Optional issue number
                
        Returns:
            Complete solution context including analysis and proposals.
        """
        self.context.issue_data = issue_data
        
        # Start the main trace span
        with self._tracer.start_span(
            "solve_issue",
            kind=SpanKind.SERVER,
            attributes={
                "issue.title": issue_data.get("title", "Unknown"),
                "issue.number": issue_data.get("number"),
                "project.path": str(self.project_path),
                "execution.mode": self._execution_mode.value,
            }
        ) as root_span:
            self._current_trace_id = root_span.trace_id
            
            # Create execution timeline
            timeline = ExecutionTimeline(trace_id=root_span.trace_id)
            timeline.metadata["issue_title"] = issue_data.get("title", "Unknown")
            timeline.metadata["execution_mode"] = self._execution_mode.value
            
            logger.info(f"Starting agentic chain for issue: {issue_data.get('title', 'Unknown')}")
            logger.info(f"Execution mode: {self._execution_mode.value}")
            if self._llm_provider:
                logger.info(f"LLM enabled: {self._llm_provider.config.provider}/{self._llm_provider.config.model}")
                root_span.set_attribute("llm.provider", self._llm_provider.config.provider)
                root_span.set_attribute("llm.model", self._llm_provider.config.model)
            
            chain_success = True
            user_cancelled = False
            for agent in self.agents:
                start_time = time.perf_counter()
                
                # Create span for each agent
                with self._tracer.start_span(
                    f"agent.{agent.name}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "agent.name": agent.name,
                        "agent.type": type(agent).__name__,
                    }
                ) as agent_span:
                    # Create agent step for timeline
                    step = AgentStep(
                        name=f"agent.{agent.name}",
                        agent_name=agent.name,
                        status="running",
                    )
                    step.start_time = agent_span.start_time
                    
                    logger.info(f"Executing agent: {agent.name}")
                    try:
                        self.context = agent.execute(self.context)
                        
                        duration = time.perf_counter() - start_time
                        agent_span.set_attribute("execution.duration_seconds", duration)
                        agent_span.set_status(SpanStatus.OK)
                        
                        # Update step
                        step.status = "success"
                        step.duration_ms = duration * 1000
                        
                        # Record metrics
                        self._metrics.record_execution_time(agent.name, duration, success=True)
                        
                        logger.info(f"Agent {agent.name} completed successfully in {duration:.3f}s")
                        
                        # Interactive mode: request review at key decision points
                        if (
                            self._interactive 
                            and self._interaction_handler is not None
                            and self._interaction_handler.enabled
                        ):
                            should_continue = self._handle_agent_interaction(agent)
                            if not should_continue:
                                user_cancelled = True
                                chain_success = False
                                logger.info("User cancelled operation in interactive mode")
                                break
                        
                    except Exception as e:
                        duration = time.perf_counter() - start_time
                        agent_span.record_exception(e)
                        
                        # Update step
                        step.status = "error"
                        step.error_message = str(e)
                        step.duration_ms = duration * 1000
                        
                        # Record metrics
                        self._metrics.record_execution_time(agent.name, duration, success=False)
                        
                        chain_success = False
                        logger.error(f"Agent {agent.name} failed: {str(e)}")
                        raise
                    finally:
                        timeline.add_step(step)
            # Execute based on mode
            if self._execution_mode == ExecutionMode.PARALLEL:
                chain_success = self._execute_parallel(timeline, root_span)
            else:
                chain_success = self._execute_sequential(timeline, root_span)
            
            # Interactive mode: final solution review
            if (
                chain_success 
                and not user_cancelled
                and self._interactive 
                and self._interaction_handler is not None
                and self._interaction_handler.enabled
            ):
                chain_success = self._handle_solution_review()
            
            # Complete interaction history
            if self._interaction_handler:
                self._interaction_handler.history.complete()
            
            # Complete timeline
            timeline.complete(success=chain_success)
            self._execution_timelines.append(timeline)
            
            # Record LLM usage in metrics
            if self._llm_provider:
                usage = self.context.llm_context
                if usage.total_tokens > 0:
                    self._metrics.record_llm_usage(
                        provider=usage.provider or "",
                        model=usage.model or "",
                        prompt_tokens=usage.total_prompt_tokens,
                        completion_tokens=usage.total_completion_tokens,
                        duration_seconds=0,  # Not tracked separately
                        cost=usage.estimated_cost,
                    )
                    root_span.set_attribute("llm.total_tokens", usage.total_tokens)
                    root_span.set_attribute("llm.estimated_cost", usage.estimated_cost)
                
        self._executed = True
        return self.get_result()
    
    def _execute_sequential(self, timeline: ExecutionTimeline, root_span) -> bool:
        """
        Execute agents sequentially (original behavior).
        
        Args:
            timeline: The execution timeline to update.
            root_span: The root tracing span.
            
        Returns:
            True if all agents completed successfully.
        """
        chain_success = True
        for agent in self.agents:
            start_time = time.perf_counter()
            
            # Create span for each agent
            with self._tracer.start_span(
                f"agent.{agent.name}",
                kind=SpanKind.INTERNAL,
                attributes={
                    "agent.name": agent.name,
                    "agent.type": type(agent).__name__,
                }
            ) as agent_span:
                # Create agent step for timeline
                step = AgentStep(
                    name=f"agent.{agent.name}",
                    agent_name=agent.name,
                    status="running",
                )
                step.start_time = agent_span.start_time
                
                logger.info(f"Executing agent: {agent.name}")
                try:
                    self.context = agent.execute(self.context)
                    
                    duration = time.perf_counter() - start_time
                    agent_span.set_attribute("execution.duration_seconds", duration)
                    agent_span.set_status(SpanStatus.OK)
                    
                    # Update step
                    step.status = "success"
                    step.duration_ms = duration * 1000
                    
                    # Record metrics
                    self._metrics.record_execution_time(agent.name, duration, success=True)
                    
                    logger.info(f"Agent {agent.name} completed successfully in {duration:.3f}s")
                    
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    agent_span.record_exception(e)
                    
                    # Update step
                    step.status = "error"
                    step.error_message = str(e)
                    step.duration_ms = duration * 1000
                    
                    # Record metrics
                    self._metrics.record_execution_time(agent.name, duration, success=False)
                    
                    chain_success = False
                    logger.error(f"Agent {agent.name} failed: {str(e)}")
                    raise
                finally:
                    timeline.add_step(step)
        
        return chain_success
    
    def _execute_parallel(self, timeline: ExecutionTimeline, root_span) -> bool:
        """
        Execute agents in parallel where dependencies allow.
        
        Args:
            timeline: The execution timeline to update.
            root_span: The root tracing span.
            
        Returns:
            True if all agents completed successfully.
        """
        # Create dependency graph with default dependencies
        graph = create_default_dependency_graph(self.agents)
        
        # Create progress callback that also updates timeline
        def progress_with_timeline(agent_name: str, status: str, progress: float):
            step = AgentStep(
                name=f"agent.{agent_name}",
                agent_name=agent_name,
                status=status,
            )
            timeline.add_step(step)
            
            # Forward to user callback if configured
            if self._progress_callback:
                self._progress_callback(agent_name, status, progress)
        
        # Create executor
        executor = ParallelExecutor(
            config=self._parallel_config,
            progress_callback=progress_with_timeline,
        )
        
        # Execute
        logger.info(f"Starting parallel execution with max_workers={self._parallel_config.max_workers}")
        result = executor.execute(graph, self.context)
        self._last_parallel_result = result
        
        # Record metrics for all agents
        for agent_name, exec_time in result.agent_times.items():
            success = agent_name in result.completed_agents
            self._metrics.record_execution_time(agent_name, exec_time, success=success)
        
        # Log results
        logger.info(f"Parallel execution completed in {result.total_execution_time:.3f}s")
        logger.info(f"Completed: {result.completed_agents}")
        if result.failed_agents:
            logger.error(f"Failed: {result.failed_agents}")
        if result.skipped_agents:
            logger.warning(f"Skipped: {result.skipped_agents}")
        
        # Add parallel execution metadata to span
        root_span.set_attribute("parallel.total_time_seconds", result.total_execution_time)
        root_span.set_attribute("parallel.completed_count", len(result.completed_agents))
        root_span.set_attribute("parallel.failed_count", len(result.failed_agents))
        root_span.set_attribute("parallel.skipped_count", len(result.skipped_agents))
        
        # Handle failures based on configuration
        if not result.success and not self._parallel_config.continue_on_partial_failure:
            failed_info = "; ".join(
                f"{f['name']}: {f['error']}" for f in result.failed_agents
            )
            raise RuntimeError(f"Parallel execution failed: {failed_info}")
        
        return result.success
    
    def analyze_project(self) -> dict:
        """
        Run only project analysis without solving an issue.
        
        Returns:
            Project analysis results.
        """
        with self._tracer.start_span("analyze_project") as span:
            analyzer = ProjectAnalyzer()
            start_time = time.perf_counter()
            self.context = analyzer.execute(self.context)
            duration = time.perf_counter() - start_time
            span.set_attribute("execution.duration_seconds", duration)
            self._metrics.record_execution_time("ProjectAnalyzer", duration, success=True)
        return self.context.project_analysis
    
    def get_result(self) -> dict:
        """
        Get the complete result of the agentic chain.
        
        Returns:
            Dictionary containing all analysis and solution data.
        """
        result = self.context.to_dict()
        
        # Add trace ID if available
        if self._current_trace_id:
            result["trace_id"] = self._current_trace_id
        
        return result
    
    def get_llm_usage(self) -> dict:
        """
        Get LLM usage statistics.
        
        Returns:
            Dictionary with usage information.
        """
        return self.context.llm_context.to_dict()
    
    def get_execution_timeline(self) -> Optional[ExecutionTimeline]:
        """
        Get the most recent execution timeline.
        
        Returns:
            ExecutionTimeline if available, None otherwise.
        """
        if self._execution_timelines:
            return self._execution_timelines[-1]
        return None
    
    def get_all_timelines(self) -> List[ExecutionTimeline]:
        """
        Get all execution timelines.
        
        Returns:
            List of all ExecutionTimelines.
        """
        return self._execution_timelines.copy()
    
    def get_observability_data(self) -> ObservabilityData:
        """
        Get aggregated observability data for dashboard display.
        
        Returns:
            ObservabilityData with metrics, timelines, and errors.
        """
        return ObservabilityData.from_collector(
            self._metrics,
            self._execution_timelines,
        )
    
    def get_trace_statistics(self) -> dict:
        """
        Get tracing statistics.
        
        Returns:
            Dictionary with trace statistics.
        """
        return self._tracer.get_statistics()
    
    def get_metrics_summary(self) -> dict:
        """
        Get metrics summary.
        
        Returns:
            Dictionary with metrics summary.
        """
        return self._metrics.get_summary()
    
    def get_solution_summary(self) -> str:
        """
        Get a human-readable summary of the solution.
        
        Returns:
            Formatted summary string.
        """
        if not self._executed:
            return "Chain has not been executed yet. Call solve_issue() first."
            
        result = self.get_result()
        
        lines = []
        lines.append("=" * 60)
        lines.append("AGENTIC CHAIN SOLUTION SUMMARY")
        lines.append("=" * 60)
        
        # Trace info
        if result.get("trace_id"):
            lines.append(f"\n🔍 Trace ID: {result['trace_id'][:16]}...")
        
        # Issue Analysis
        if result.get("issue_analysis"):
            analysis = result["issue_analysis"]
            lines.append("\n📋 ISSUE ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Title: {analysis.get('title', 'N/A')}")
            lines.append(f"Type: {analysis.get('issue_type', 'unknown')}")
            lines.append(f"Priority: {analysis.get('priority', 'medium')}")
            
            if analysis.get("requirements"):
                lines.append("\nRequirements:")
                for req in analysis["requirements"][:5]:
                    lines.append(f"  • {req[:80]}")
                    
        # Code Review
        if result.get("code_review"):
            review = result["code_review"]
            lines.append("\n🔍 CODE REVIEW")
            lines.append("-" * 40)
            
            if review.get("relevant_files"):
                lines.append(f"Relevant files: {len(review['relevant_files'])}")
                for f in review["relevant_files"][:5]:
                    lines.append(f"  • {f}")
                    
            if review.get("potential_issues"):
                lines.append(f"\nPotential issues found: {len(review['potential_issues'])}")
                
        # Solution
        if result.get("solution"):
            solution = result["solution"]
            lines.append("\n💡 PROPOSED SOLUTION")
            lines.append("-" * 40)
            
            # Show if LLM was used
            if solution.get("llm_generated"):
                lines.append("✨ AI-generated solution")
            
            if solution.get("implementation_plan"):
                plan = solution["implementation_plan"]
                lines.append(f"Complexity: {plan.get('complexity', 'N/A')}")
                lines.append(f"Estimated hours: {plan.get('estimated_hours', 'N/A')}")
                
            if solution.get("proposed_changes"):
                lines.append("\nProposed changes:")
                for change in solution["proposed_changes"][:3]:
                    lines.append(f"  • [{change.get('type', 'change')}] {change.get('description', '')[:60]}")
                    
            if solution.get("risks"):
                lines.append("\nRisks:")
                for risk in solution["risks"]:
                    lines.append(f"  • [{risk.get('level', 'N/A')}] {risk.get('description', '')[:60]}")
        
        # LLM Usage
        llm_usage = result.get("llm_usage", {})
        if llm_usage.get("total_tokens", 0) > 0:
            lines.append("\n🤖 LLM USAGE")
            lines.append("-" * 40)
            lines.append(f"Provider: {llm_usage.get('provider', 'N/A')}")
            lines.append(f"Model: {llm_usage.get('model', 'N/A')}")
            lines.append(f"Total tokens: {llm_usage.get('total_tokens', 0)}")
            lines.append(f"Estimated cost: ${llm_usage.get('estimated_cost', 0):.4f}")
        
        # Execution metrics
        timeline = self.get_execution_timeline()
        if timeline:
            lines.append("\n📊 EXECUTION METRICS")
            lines.append("-" * 40)
            lines.append(f"Total duration: {timeline.total_duration_ms:.2f}ms")
            lines.append(f"Steps: {timeline.step_count} ({timeline.success_count} successful, {timeline.error_count} errors)")
                    
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def export_result(self, output_path: str):
        """
        Export the result to a JSON file.
        
        Args:
            output_path: Path to save the JSON output.
        """
        result = self.get_result()
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
        logger.info(f"Result exported to {output_path}")
    
    def export_traces(self, output_path: str):
        """
        Export all traces to a JSON file.
        
        Args:
            output_path: Path to save the traces.
        """
        spans = [span.to_dict() for span in self._tracer.spans]
        with open(output_path, 'w') as f:
            json.dump({"spans": spans}, f, indent=2, default=str)
        logger.info(f"Traces exported to {output_path}")
        
    def __repr__(self) -> str:
        agent_names = [a.name for a in self.agents]
        llm_info = ""
        if self._llm_provider:
            llm_info = f", llm={self._llm_provider.config.provider}"
        tracing_info = f", tracing={'enabled' if self._tracer.config.enabled else 'disabled'}"
        interactive_info = f", interactive={'enabled' if self._interactive else 'disabled'}"
        return f"AgenticChain(project='{self.project_path}', agents={agent_names}{llm_info}{tracing_info}{interactive_info})"
