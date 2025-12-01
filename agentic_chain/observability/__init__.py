"""
Observability module for Agentic Chain - provides tracing, metrics, and logging.

This module provides comprehensive observability capabilities for monitoring
and debugging agent execution, including:
- OpenTelemetry-based distributed tracing
- Structured logging with context
- Metrics collection (execution time, success rate, etc.)
- Export to common tools (Jaeger, Prometheus)
"""

from .tracer import (
    Tracer,
    Span,
    SpanStatus,
    SpanKind,
    TracerConfig,
)
from .metrics import (
    MetricsCollector,
    MetricType,
    Metric,
)
from .logging import (
    StructuredLogger,
    LogLevel,
)
from .context import (
    TraceContext,
    ContextManager,
)
from .exporters import (
    Exporter,
    ConsoleExporter,
    JSONExporter,
    PrometheusExporter,
    JaegerExporter,
)
from .dashboard import (
    ObservabilityData,
    ExecutionTimeline,
    AgentStep,
)

__all__ = [
    # Tracer
    "Tracer",
    "Span",
    "SpanStatus",
    "SpanKind",
    "TracerConfig",
    # Metrics
    "MetricsCollector",
    "MetricType",
    "Metric",
    # Logging
    "StructuredLogger",
    "LogLevel",
    # Context
    "TraceContext",
    "ContextManager",
    # Exporters
    "Exporter",
    "ConsoleExporter",
    "JSONExporter",
    "PrometheusExporter",
    "JaegerExporter",
    # Dashboard
    "ObservabilityData",
    "ExecutionTimeline",
    "AgentStep",
]
