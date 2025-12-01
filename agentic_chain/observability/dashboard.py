"""
Dashboard data structures for observability visualization.

Provides data structures for execution timeline, agent steps,
and aggregated observability data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .tracer import Span, SpanStatus
from .metrics import MetricsCollector


@dataclass
class AgentStep:
    """
    Represents a single step in agent execution.
    
    Used for building execution timelines and visualizations.
    """
    
    name: str
    agent_name: str
    status: str = "pending"  # pending, running, success, error
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_summary: str = ""
    output_summary: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_span(cls, span: Span) -> "AgentStep":
        """Create an AgentStep from a Span."""
        status = "success" if span.status == SpanStatus.OK else "error" if span.status == SpanStatus.ERROR else "pending"
        
        return cls(
            name=span.name,
            agent_name=span.attributes.get("agent.name", span.name),
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            duration_ms=span.duration_ms,
            error_message=span.status_message if span.status == SpanStatus.ERROR else None,
            metadata=span.attributes,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "agent_name": self.agent_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionTimeline:
    """
    Timeline of agent execution steps.
    
    Provides a chronological view of all steps in an agent chain execution.
    """
    
    trace_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, success, error
    steps: List[AgentStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration_ms(self) -> Optional[float]:
        """Get total execution duration."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    @property
    def step_count(self) -> int:
        """Get number of steps."""
        return len(self.steps)
    
    @property
    def success_count(self) -> int:
        """Get number of successful steps."""
        return sum(1 for s in self.steps if s.status == "success")
    
    @property
    def error_count(self) -> int:
        """Get number of failed steps."""
        return sum(1 for s in self.steps if s.status == "error")
    
    def add_step(self, step: AgentStep) -> None:
        """Add a step to the timeline."""
        self.steps.append(step)
    
    def complete(self, success: bool = True) -> None:
        """Mark the timeline as complete."""
        self.end_time = datetime.now()
        self.status = "success" if success else "error"
    
    @classmethod
    def from_spans(cls, spans: List[Span], trace_id: str) -> "ExecutionTimeline":
        """Create timeline from spans."""
        if not spans:
            return cls(trace_id=trace_id)
        
        # Sort spans by start time
        sorted_spans = sorted(spans, key=lambda s: s.start_time)
        
        # Create timeline
        timeline = cls(
            trace_id=trace_id,
            start_time=sorted_spans[0].start_time,
        )
        
        # Add steps
        for span in sorted_spans:
            timeline.add_step(AgentStep.from_span(span))
        
        # Check if any spans have ended
        ended_spans = [s for s in sorted_spans if s.end_time]
        if ended_spans:
            latest_end = max(s.end_time for s in ended_spans)
            timeline.end_time = latest_end
            
            # Determine overall status
            has_error = any(s.status == SpanStatus.ERROR for s in sorted_spans)
            timeline.status = "error" if has_error else "success"
        
        return timeline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "step_count": self.step_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }


@dataclass
class ObservabilityData:
    """
    Aggregated observability data for dashboard display.
    
    Combines traces, metrics, and execution timelines for
    comprehensive system visibility.
    """
    
    # Summary statistics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    
    # LLM usage
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    
    # Recent timelines
    recent_timelines: List[ExecutionTimeline] = field(default_factory=list)
    
    # Metrics snapshot
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Error summary
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamp
    generated_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_collector(
        cls,
        collector: MetricsCollector,
        timelines: Optional[List[ExecutionTimeline]] = None,
    ) -> "ObservabilityData":
        """Create observability data from metrics collector."""
        summary = collector.get_summary()
        
        # Extract recent errors from timelines
        recent_errors = []
        if timelines:
            for timeline in timelines:
                for step in timeline.steps:
                    if step.status == "error" and step.error_message:
                        recent_errors.append({
                            "trace_id": timeline.trace_id,
                            "agent": step.agent_name,
                            "error": step.error_message,
                            "timestamp": step.start_time.isoformat() if step.start_time else None,
                        })
        
        return cls(
            total_executions=summary.get("total_executions", 0),
            successful_executions=summary.get("total_executions", 0) - summary.get("total_errors", 0),
            failed_executions=summary.get("total_errors", 0),
            avg_execution_time_ms=summary.get("avg_execution_time_seconds", 0) * 1000,
            p95_execution_time_ms=summary.get("p95_execution_time_seconds", 0) * 1000,
            total_llm_tokens=summary.get("total_llm_tokens", 0),
            total_llm_cost=summary.get("total_llm_cost", 0),
            recent_timelines=timelines[:10] if timelines else [],
            metrics_snapshot=collector.get_all_metrics(),
            recent_errors=recent_errors[:20],
        )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_executions": self.total_executions,
                "successful_executions": self.successful_executions,
                "failed_executions": self.failed_executions,
                "success_rate": round(self.success_rate, 2),
                "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
                "p95_execution_time_ms": round(self.p95_execution_time_ms, 2),
            },
            "llm_usage": {
                "total_tokens": self.total_llm_tokens,
                "total_cost": round(self.total_llm_cost, 4),
            },
            "recent_timelines": [t.to_dict() for t in self.recent_timelines],
            "metrics": self.metrics_snapshot,
            "recent_errors": self.recent_errors,
            "generated_at": self.generated_at.isoformat(),
        }
