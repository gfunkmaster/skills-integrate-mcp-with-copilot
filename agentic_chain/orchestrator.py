"""
Orchestrator - Chains agents together to solve issues.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

from .agents import AgentContext, BaseAgent
from .agents.project_analyzer import ProjectAnalyzer
from .agents.issue_analyzer import IssueAnalyzer
from .agents.code_reviewer import CodeReviewer
from .agents.solution_implementer import SolutionImplementer
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


logger = logging.getLogger(__name__)


class AgenticChain:
    """
    Orchestrates a chain of agents to analyze projects, understand issues,
    review code, and propose solutions.
    
    Supports LLM integration for intelligent code generation.
    Includes comprehensive observability with tracing, metrics, and logging.
    
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
    """
    
    def __init__(
        self,
        project_path: str,
        custom_agents: Optional[list] = None,
        llm_provider: Optional["LLMProvider"] = None,
        llm_config: Optional[dict] = None,
        enable_tracing: bool = True,
        tracer_config: Optional[TracerConfig] = None,
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
        """
        self.project_path = Path(project_path).resolve()
        self.context = AgentContext(project_path=str(self.project_path))
        
        # Set up observability
        self._tracer = Tracer(tracer_config or TracerConfig(enabled=enable_tracing))
        self._metrics = MetricsCollector()
        self._current_trace_id: Optional[str] = None
        self._execution_timelines: List[ExecutionTimeline] = []
        
        # Set up LLM provider
        self._llm_provider = llm_provider
        if not self._llm_provider and llm_config:
            self._llm_provider = self._create_llm_from_config(llm_config)
        
        # Create solution implementer with LLM if available
        solution_implementer = SolutionImplementer()
        if self._llm_provider:
            solution_implementer.llm_provider = self._llm_provider
        
        # Default agent pipeline
        self.agents = [
            ProjectAnalyzer(),
            IssueAnalyzer(),
            CodeReviewer(),
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
        # Update SolutionImplementer with new provider
        for agent in self.agents:
            if isinstance(agent, SolutionImplementer):
                agent.llm_provider = provider
    
    @property
    def tracer(self) -> Tracer:
        """Get the tracer instance."""
        return self._tracer
    
    @property
    def metrics(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._metrics
        
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
            }
        ) as root_span:
            self._current_trace_id = root_span.trace_id
            
            # Create execution timeline
            timeline = ExecutionTimeline(trace_id=root_span.trace_id)
            timeline.metadata["issue_title"] = issue_data.get("title", "Unknown")
            
            logger.info(f"Starting agentic chain for issue: {issue_data.get('title', 'Unknown')}")
            if self._llm_provider:
                logger.info(f"LLM enabled: {self._llm_provider.config.provider}/{self._llm_provider.config.model}")
                root_span.set_attribute("llm.provider", self._llm_provider.config.provider)
                root_span.set_attribute("llm.model", self._llm_provider.config.model)
            
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
        return f"AgenticChain(project='{self.project_path}', agents={agent_names}{llm_info}{tracing_info})"
