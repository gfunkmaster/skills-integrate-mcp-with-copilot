"""
Tracing context management for observability.

Provides context propagation for distributed tracing.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar


# Context variable for trace propagation
_current_context: ContextVar["TraceContext"] = ContextVar("current_trace_context")


@dataclass
class TraceContext:
    """
    Context for distributed tracing across agent executions.
    
    Attributes:
        trace_id: Unique identifier for the entire trace
        span_id: Unique identifier for the current span
        parent_span_id: ID of the parent span, if any
        start_time: When this context was created
        attributes: Additional context attributes
        baggage: Key-value pairs propagated across boundaries
    """
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    attributes: Dict[str, Any] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def create_child(self) -> "TraceContext":
        """Create a child context for nested spans."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            attributes=self.attributes.copy(),
            baggage=self.baggage.copy(),
        )
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on this context."""
        self.attributes[key] = value
    
    def set_baggage(self, key: str, value: str) -> None:
        """Set a baggage item for cross-boundary propagation."""
        self.baggage[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "attributes": self.attributes,
            "baggage": self.baggage,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Create context from dictionary."""
        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            span_id=data.get("span_id", str(uuid.uuid4())[:16]),
            parent_span_id=data.get("parent_span_id"),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else datetime.now(),
            attributes=data.get("attributes", {}),
            baggage=data.get("baggage", {}),
        )


class ContextManager:
    """
    Manages trace context across agent executions.
    
    Provides thread-safe context management using context variables.
    """
    
    @staticmethod
    def get_current() -> Optional[TraceContext]:
        """Get the current trace context."""
        try:
            return _current_context.get()
        except LookupError:
            return None
    
    @staticmethod
    def set_current(context: TraceContext) -> None:
        """Set the current trace context."""
        _current_context.set(context)
    
    @staticmethod
    def create_context(**attributes) -> TraceContext:
        """Create a new trace context with optional attributes."""
        context = TraceContext()
        for key, value in attributes.items():
            context.set_attribute(key, value)
        return context
    
    @staticmethod
    def create_child_context() -> TraceContext:
        """Create a child context from the current context."""
        current = ContextManager.get_current()
        if current:
            return current.create_child()
        return TraceContext()
    
    @staticmethod
    def clear() -> None:
        """Clear the current trace context."""
        try:
            _current_context.set(TraceContext())
        except ValueError:
            pass


class ContextScope:
    """
    Context manager for scoped trace context.
    
    Usage:
        with ContextScope(context):
            # context is active here
            pass
        # previous context is restored
    """
    
    def __init__(self, context: TraceContext):
        self.context = context
        self.previous: Optional[TraceContext] = None
    
    def __enter__(self) -> TraceContext:
        self.previous = ContextManager.get_current()
        ContextManager.set_current(self.context)
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous:
            ContextManager.set_current(self.previous)
        else:
            ContextManager.clear()
        return False
