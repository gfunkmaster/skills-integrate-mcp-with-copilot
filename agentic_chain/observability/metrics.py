"""
Metrics collection for observability.

Provides counters, gauges, histograms, and other metric types
for tracking agent performance and health.
"""

import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps


class MetricType(str, Enum):
    """Type of metric."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """
    Represents a metric being tracked.
    
    Supports counters, gauges, histograms, and summaries.
    """
    
    name: str
    type: MetricType
    description: str = ""
    unit: str = ""
    values: List[MetricValue] = field(default_factory=list)
    
    # For histograms
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        self.values.append(MetricValue(
            value=value,
            labels=labels or {},
        ))
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value."""
        if self.values:
            return self.values[-1].value
        return None
    
    def get_sum(self) -> float:
        """Get sum of all values (for counters)."""
        return sum(v.value for v in self.values)
    
    def get_average(self) -> Optional[float]:
        """Get average of all values."""
        if self.values:
            return statistics.mean(v.value for v in self.values)
        return None
    
    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get a percentile value (0-100)."""
        if not self.values:
            return None
        sorted_values = sorted(v.value for v in self.values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_histogram_buckets(self) -> Dict[str, int]:
        """Get histogram bucket counts."""
        bucket_counts = {f"le_{b}": 0 for b in self.buckets}
        bucket_counts["le_inf"] = 0
        
        for metric_value in self.values:
            value = metric_value.value
            for bucket in self.buckets:
                if value <= bucket:
                    bucket_counts[f"le_{bucket}"] += 1
            bucket_counts["le_inf"] += 1
        
        return bucket_counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        base = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "unit": self.unit,
            "value_count": len(self.values),
        }
        
        if self.type == MetricType.COUNTER:
            base["sum"] = self.get_sum()
        elif self.type == MetricType.GAUGE:
            base["current"] = self.get_current_value()
        elif self.type == MetricType.HISTOGRAM:
            base["buckets"] = self.get_histogram_buckets()
            base["average"] = self.get_average()
            base["p50"] = self.get_percentile(50)
            base["p95"] = self.get_percentile(95)
            base["p99"] = self.get_percentile(99)
        elif self.type == MetricType.SUMMARY:
            base["average"] = self.get_average()
            base["p50"] = self.get_percentile(50)
            base["p95"] = self.get_percentile(95)
            base["p99"] = self.get_percentile(99)
        
        return base
    
    def to_prometheus(self) -> str:
        """Export metric in Prometheus text format."""
        lines = []
        
        # Add help and type
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} {self.type.value}")
        
        if self.type == MetricType.COUNTER:
            lines.append(f"{self.name}_total {self.get_sum()}")
        elif self.type == MetricType.GAUGE:
            value = self.get_current_value()
            if value is not None:
                lines.append(f"{self.name} {value}")
        elif self.type == MetricType.HISTOGRAM:
            buckets = self.get_histogram_buckets()
            for bucket_name, count in buckets.items():
                le_value = bucket_name.replace("le_", "").replace("inf", "+Inf")
                lines.append(f'{self.name}_bucket{{le="{le_value}"}} {count}')
            lines.append(f"{self.name}_count {len(self.values)}")
            lines.append(f"{self.name}_sum {self.get_sum()}")
        elif self.type == MetricType.SUMMARY:
            for percentile in [50, 90, 95, 99]:
                value = self.get_percentile(percentile)
                if value is not None:
                    lines.append(f'{self.name}{{quantile="{percentile / 100}"}} {value}')
            lines.append(f"{self.name}_count {len(self.values)}")
            lines.append(f"{self.name}_sum {self.get_sum()}")
        
        return "\n".join(lines)


class MetricsCollector:
    """
    Central collector for all metrics.
    
    Provides methods to create and record various metric types,
    track agent performance, and export to monitoring systems.
    
    Usage:
        collector = MetricsCollector()
        
        # Record execution time
        collector.record_execution_time("IssueAnalyzer", 0.5)
        
        # Increment counter
        collector.increment_counter("agent_executions", labels={"agent": "IssueAnalyzer"})
        
        # Get all metrics
        metrics = collector.get_all_metrics()
    """
    
    def __init__(self, prefix: str = "agentic_chain"):
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._setup_default_metrics()
    
    def _setup_default_metrics(self) -> None:
        """Set up default metrics for agent monitoring."""
        # Execution metrics
        self._create_metric(
            "agent_execution_duration_seconds",
            MetricType.HISTOGRAM,
            "Agent execution duration in seconds",
            "seconds",
        )
        self._create_metric(
            "agent_executions_total",
            MetricType.COUNTER,
            "Total number of agent executions",
        )
        self._create_metric(
            "agent_errors_total",
            MetricType.COUNTER,
            "Total number of agent execution errors",
        )
        
        # LLM metrics
        self._create_metric(
            "llm_tokens_total",
            MetricType.COUNTER,
            "Total LLM tokens used",
            "tokens",
        )
        self._create_metric(
            "llm_request_duration_seconds",
            MetricType.HISTOGRAM,
            "LLM API request duration in seconds",
            "seconds",
        )
        self._create_metric(
            "llm_cost_total",
            MetricType.COUNTER,
            "Total estimated LLM cost",
            "USD",
        )
        
        # Memory metrics
        self._create_metric(
            "memory_usage_bytes",
            MetricType.GAUGE,
            "Current memory usage in bytes",
            "bytes",
        )
        
        # Success rate
        self._create_metric(
            "success_rate",
            MetricType.GAUGE,
            "Current success rate",
            "percent",
        )
    
    def _create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
    ) -> Metric:
        """Create a new metric."""
        full_name = f"{self.prefix}_{name}"
        metric = Metric(
            name=full_name,
            type=metric_type,
            description=description,
            unit=unit,
        )
        self._metrics[full_name] = metric
        return metric
    
    def get_or_create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
    ) -> Metric:
        """Get an existing metric or create a new one."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            return self._create_metric(name, metric_type, description, unit)
        return self._metrics[full_name]
    
    def record_execution_time(
        self,
        agent_name: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record agent execution time."""
        metric = self._metrics.get(f"{self.prefix}_agent_execution_duration_seconds")
        if metric:
            metric.record(duration_seconds, {"agent": agent_name})
        
        # Increment execution counter
        executions = self._metrics.get(f"{self.prefix}_agent_executions_total")
        if executions:
            executions.record(1, {"agent": agent_name, "status": "success" if success else "error"})
        
        if not success:
            errors = self._metrics.get(f"{self.prefix}_agent_errors_total")
            if errors:
                errors.record(1, {"agent": agent_name})
    
    def record_llm_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
        cost: float = 0.0,
    ) -> None:
        """Record LLM API usage metrics."""
        labels = {"provider": provider, "model": model}
        
        # Token count
        tokens = self._metrics.get(f"{self.prefix}_llm_tokens_total")
        if tokens:
            tokens.record(prompt_tokens + completion_tokens, labels)
        
        # Duration
        duration = self._metrics.get(f"{self.prefix}_llm_request_duration_seconds")
        if duration:
            duration.record(duration_seconds, labels)
        
        # Cost
        cost_metric = self._metrics.get(f"{self.prefix}_llm_cost_total")
        if cost_metric:
            cost_metric.record(cost, labels)
    
    def record_memory_usage(self, bytes_used: int) -> None:
        """Record current memory usage."""
        metric = self._metrics.get(f"{self.prefix}_memory_usage_bytes")
        if metric:
            metric.record(bytes_used)
    
    def record_success_rate(self, rate: float) -> None:
        """Record current success rate (0-100)."""
        metric = self._metrics.get(f"{self.prefix}_success_rate")
        if metric:
            metric.record(rate)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        metric = self.get_or_create_metric(name, MetricType.COUNTER)
        metric.record(value, labels)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        metric = self.get_or_create_metric(name, MetricType.GAUGE)
        metric.record(value, labels)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        metric = self.get_or_create_metric(name, MetricType.HISTOGRAM)
        metric.record(value, labels)
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        full_name = f"{self.prefix}_{name}"
        return self._metrics.get(full_name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionaries."""
        return {name: metric.to_dict() for name, metric in self._metrics.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        execution_metric = self._metrics.get(f"{self.prefix}_agent_execution_duration_seconds")
        executions = self._metrics.get(f"{self.prefix}_agent_executions_total")
        errors = self._metrics.get(f"{self.prefix}_agent_errors_total")
        llm_tokens = self._metrics.get(f"{self.prefix}_llm_tokens_total")
        llm_cost = self._metrics.get(f"{self.prefix}_llm_cost_total")
        
        total_executions = executions.get_sum() if executions else 0
        total_errors = errors.get_sum() if errors else 0
        
        return {
            "total_executions": int(total_executions),
            "total_errors": int(total_errors),
            "success_rate": (total_executions - total_errors) / total_executions * 100 if total_executions > 0 else 0,
            "avg_execution_time_seconds": execution_metric.get_average() if execution_metric else 0,
            "p95_execution_time_seconds": execution_metric.get_percentile(95) if execution_metric else 0,
            "total_llm_tokens": int(llm_tokens.get_sum()) if llm_tokens else 0,
            "total_llm_cost": llm_cost.get_sum() if llm_cost else 0,
        }
    
    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        for metric in self._metrics.values():
            lines.append(metric.to_prometheus())
            lines.append("")
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all metric values."""
        for metric in self._metrics.values():
            metric.values.clear()


# Timing context manager
class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time: float = 0
        self.duration: float = 0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.perf_counter() - self.start_time
        self.collector.record_histogram(self.metric_name, self.duration, self.labels)
        return False


# Decorator for timing functions
F = TypeVar('F', bound=Callable[..., Any])


def timed(
    collector: MetricsCollector,
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[F], F]:
    """Decorator to time a function and record to metrics."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(collector, metric_name, labels):
                return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator
