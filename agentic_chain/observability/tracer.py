"""
Distributed tracing for agent execution.

Provides OpenTelemetry-compatible tracing with spans for tracking
agent execution flow and performance.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps

from .context import TraceContext, ContextManager, ContextScope


class SpanStatus(str, Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanKind(str, Enum):
    """Kind of span for semantic meaning."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanEvent:
    """An event that occurred during a span."""
    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """
    Represents a unit of work or operation being traced.
    
    Spans track the execution of a single operation, capturing timing,
    status, and contextual information.
    """
    
    name: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    
    # Computed properties
    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds."""
        if self.end_time and self.start_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None
    
    @property
    def is_ended(self) -> bool:
        """Check if span has ended."""
        return self.end_time is not None
    
    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on this span."""
        self.attributes[key] = value
        return self
    
    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes at once."""
        self.attributes.update(attributes)
        return self
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add an event to this span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))
        return self
    
    def set_status(
        self,
        status: SpanStatus,
        message: Optional[str] = None,
    ) -> "Span":
        """Set the status of this span."""
        self.status = status
        self.status_message = message
        return self
    
    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Record an exception as a span event."""
        event_attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        }
        if attributes:
            event_attrs.update(attributes)
        
        self.add_event("exception", event_attrs)
        self.set_status(SpanStatus.ERROR, str(exception))
        return self
    
    def end(self) -> "Span":
        """End this span."""
        if not self.is_ended:
            self.end_time = datetime.now()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
        }
    
    def __enter__(self) -> "Span":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.record_exception(exc_val)
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()
        return False


@dataclass
class TracerConfig:
    """Configuration for the tracer."""
    
    service_name: str = "agentic-chain"
    enabled: bool = True
    sample_rate: float = 1.0  # 0.0 to 1.0
    max_spans: int = 1000  # Max spans to keep in memory
    export_on_end: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "service_name": self.service_name,
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "max_spans": self.max_spans,
            "export_on_end": self.export_on_end,
        }


class Tracer:
    """
    Main tracer for creating and managing spans.
    
    Provides OpenTelemetry-compatible tracing functionality for
    tracking agent execution flow.
    
    Usage:
        tracer = Tracer()
        
        with tracer.start_span("agent_execution") as span:
            span.set_attribute("agent.name", "IssueAnalyzer")
            # do work
            span.add_event("analysis_complete")
    """
    
    def __init__(self, config: Optional[TracerConfig] = None):
        self.config = config or TracerConfig()
        self._spans: List[Span] = []
        self._exporters: List[Callable[[Span], None]] = []
        self._active_span: Optional[Span] = None
    
    @property
    def spans(self) -> List[Span]:
        """Get all recorded spans."""
        return self._spans.copy()
    
    def add_exporter(self, exporter: Callable[[Span], None]) -> None:
        """Add an exporter to be called when spans end."""
        self._exporters.append(exporter)
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Name of the span
            kind: Kind of span
            attributes: Initial attributes
            parent: Parent span (auto-detected if not provided)
            
        Returns:
            New Span instance
        """
        if not self.config.enabled:
            return Span(name=name)
        
        # Get parent span info
        context = ContextManager.get_current()
        parent_id = None
        trace_id = str(uuid.uuid4())
        
        if parent:
            parent_id = parent.span_id
            trace_id = parent.trace_id
        elif context:
            parent_id = context.span_id
            trace_id = context.trace_id
        elif self._active_span:
            parent_id = self._active_span.span_id
            trace_id = self._active_span.trace_id
        
        span = Span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_id,
            kind=kind,
            attributes=attributes or {},
        )
        
        # Set service name
        span.set_attribute("service.name", self.config.service_name)
        
        return _TracedSpan(span, self)
    
    def _record_span(self, span: Span) -> None:
        """Record a completed span."""
        if len(self._spans) >= self.config.max_spans:
            # Remove oldest spans
            self._spans = self._spans[-(self.config.max_spans // 2):]
        
        self._spans.append(span)
        
        # Export to registered exporters
        if self.config.export_on_end:
            for exporter in self._exporters:
                try:
                    exporter(span)
                except Exception:
                    pass  # Don't let export failures break tracing
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a specific trace."""
        return [s for s in self._spans if s.trace_id == trace_id]
    
    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        if not self._spans:
            return {
                "total_spans": 0,
                "avg_duration_ms": 0,
                "error_count": 0,
                "success_rate": 0,
            }
        
        durations = [s.duration_ms for s in self._spans if s.duration_ms]
        errors = [s for s in self._spans if s.status == SpanStatus.ERROR]
        
        return {
            "total_spans": len(self._spans),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "error_count": len(errors),
            "success_rate": (len(self._spans) - len(errors)) / len(self._spans) if self._spans else 0,
            "unique_traces": len(set(s.trace_id for s in self._spans)),
        }


class _TracedSpan(Span):
    """Span wrapper that records to tracer on end."""
    
    def __init__(self, span: Span, tracer: Tracer):
        # Copy all attributes from the span
        self.name = span.name
        self.trace_id = span.trace_id
        self.span_id = span.span_id
        self.parent_span_id = span.parent_span_id
        self.kind = span.kind
        self.status = span.status
        self.status_message = span.status_message
        self.start_time = span.start_time
        self.end_time = span.end_time
        self.attributes = span.attributes
        self.events = span.events
        self._tracer = tracer
        self._previous_active: Optional[Span] = None
    
    def __enter__(self) -> "Span":
        self._previous_active = self._tracer._active_span
        self._tracer._active_span = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.record_exception(exc_val)
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()
        self._tracer._record_span(self)
        self._tracer._active_span = self._previous_active
        return False


# Decorator for tracing functions
F = TypeVar('F', bound=Callable[..., Any])


def trace(
    tracer: Tracer,
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.
    
    Usage:
        @trace(tracer, "my_function")
        def my_function():
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with tracer.start_span(span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        return wrapper  # type: ignore
    return decorator
