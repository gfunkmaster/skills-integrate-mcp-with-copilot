"""
Exporters for observability data.

Provides exporters for traces, metrics, and logs to various backends:
- Console (text output)
- JSON (file or stdout)
- Prometheus (metrics)
- Jaeger (traces)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
import sys

from .tracer import Span, SpanStatus
from .metrics import Metric, MetricsCollector


class Exporter(ABC):
    """Base class for exporters."""
    
    @abstractmethod
    def export_span(self, span: Span) -> None:
        """Export a single span."""
        pass
    
    @abstractmethod
    def export_spans(self, spans: List[Span]) -> None:
        """Export multiple spans."""
        pass
    
    def flush(self) -> None:
        """Flush any buffered data."""
        pass
    
    def shutdown(self) -> None:
        """Clean up resources."""
        pass


class ConsoleExporter(Exporter):
    """
    Exports spans to console in human-readable format.
    
    Useful for development and debugging.
    """
    
    def __init__(self, output: Optional[TextIO] = None, colored: bool = True):
        self.output = output or sys.stdout
        self.colored = colored
    
    def _colorize(self, text: str, color: str) -> str:
        """Add ANSI color codes."""
        if not self.colored:
            return text
        
        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "reset": "\033[0m",
        }
        return f"{colors.get(color, '')}{text}{colors.get('reset', '')}"
    
    def export_span(self, span: Span) -> None:
        """Export a span to console."""
        status_color = "green" if span.status == SpanStatus.OK else "red" if span.status == SpanStatus.ERROR else "yellow"
        status_text = self._colorize(span.status.value.upper(), status_color)
        
        duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "N/A"
        
        output_lines = [
            f"{'─' * 60}",
            f"Span: {self._colorize(span.name, 'blue')}",
            f"  Trace ID:  {span.trace_id[:16]}...",
            f"  Span ID:   {span.span_id}",
            f"  Status:    {status_text}",
            f"  Duration:  {duration}",
        ]
        
        if span.parent_span_id:
            output_lines.append(f"  Parent:    {span.parent_span_id}")
        
        if span.attributes:
            output_lines.append("  Attributes:")
            for key, value in span.attributes.items():
                output_lines.append(f"    {key}: {value}")
        
        if span.events:
            output_lines.append("  Events:")
            for event in span.events:
                output_lines.append(f"    - {event.name} @ {event.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        
        if span.status_message:
            output_lines.append(f"  Message: {span.status_message}")
        
        self.output.write("\n".join(output_lines) + "\n")
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export multiple spans."""
        for span in spans:
            self.export_span(span)


class JSONExporter(Exporter):
    """
    Exports spans as JSON.
    
    Can output to file or stdout.
    """
    
    def __init__(
        self,
        output_file: Optional[str] = None,
        output: Optional[TextIO] = None,
        pretty: bool = True,
    ):
        self.output_file = output_file
        self.output = output or sys.stdout
        self.pretty = pretty
        self._buffer: List[Dict[str, Any]] = []
    
    def export_span(self, span: Span) -> None:
        """Export a span as JSON."""
        span_dict = span.to_dict()
        
        if self.output_file:
            self._buffer.append(span_dict)
        else:
            indent = 2 if self.pretty else None
            self.output.write(json.dumps(span_dict, indent=indent) + "\n")
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export multiple spans as JSON."""
        for span in spans:
            self.export_span(span)
    
    def flush(self) -> None:
        """Flush buffered spans to file."""
        if self.output_file and self._buffer:
            with open(self.output_file, 'w') as f:
                indent = 2 if self.pretty else None
                json.dump(self._buffer, f, indent=indent)
            self._buffer.clear()


class PrometheusExporter(Exporter):
    """
    Exports metrics in Prometheus format.
    
    Provides an endpoint for Prometheus scraping.
    """
    
    def __init__(
        self,
        collector: Optional[MetricsCollector] = None,
        port: int = 9090,
        path: str = "/metrics",
    ):
        self.collector = collector or MetricsCollector()
        self.port = port
        self.path = path
        self._server = None
    
    def export_span(self, span: Span) -> None:
        """Convert span to Prometheus metrics."""
        # Record span duration as histogram
        if span.duration_ms:
            self.collector.record_histogram(
                "span_duration_seconds",
                span.duration_ms / 1000,
                {"span_name": span.name, "status": span.status.value},
            )
        
        # Increment span counter
        self.collector.increment_counter(
            "spans_total",
            1,
            {"span_name": span.name, "status": span.status.value},
        )
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export multiple spans."""
        for span in spans:
            self.export_span(span)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus text format."""
        return self.collector.to_prometheus()
    
    def start_server(self) -> None:
        """Start HTTP server for metrics endpoint."""
        # Simple implementation - in production use prometheus_client
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        exporter = self
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == exporter.path:
                    metrics = exporter.get_metrics()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(metrics.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        self._server = HTTPServer(("", self.port), MetricsHandler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
    
    def shutdown(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()


@dataclass
class JaegerSpan:
    """Jaeger-compatible span format."""
    
    trace_id: str
    span_id: str
    operation_name: str
    references: List[Dict[str, str]] = field(default_factory=list)
    flags: int = 1
    start_time: int = 0  # microseconds
    duration: int = 0  # microseconds
    tags: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    process_id: str = "p1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Jaeger JSON format."""
        return {
            "traceID": self.trace_id.replace("-", "")[:32],
            "spanID": self.span_id.replace("-", "")[:16],
            "operationName": self.operation_name,
            "references": self.references,
            "flags": self.flags,
            "startTime": self.start_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "processID": self.process_id,
        }


class JaegerExporter(Exporter):
    """
    Exports traces in Jaeger format.
    
    Can output to file or send to Jaeger collector.
    """
    
    def __init__(
        self,
        service_name: str = "agentic-chain",
        collector_endpoint: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        self.service_name = service_name
        self.collector_endpoint = collector_endpoint or os.environ.get("JAEGER_ENDPOINT")
        self.output_file = output_file
        self._traces: Dict[str, List[JaegerSpan]] = {}
    
    def _convert_span(self, span: Span) -> JaegerSpan:
        """Convert internal span to Jaeger format."""
        # Convert tags
        tags = []
        for key, value in span.attributes.items():
            tag_type = "string"
            if isinstance(value, bool):
                tag_type = "bool"
            elif isinstance(value, (int, float)):
                tag_type = "float64" if isinstance(value, float) else "int64"
            tags.append({"key": key, "type": tag_type, "value": value})
        
        # Add status tag
        tags.append({"key": "otel.status_code", "type": "string", "value": span.status.value})
        if span.status_message:
            tags.append({"key": "otel.status_description", "type": "string", "value": span.status_message})
        
        # Convert events to logs
        logs = []
        for event in span.events:
            event_fields = [{"key": "event", "type": "string", "value": event.name}]
            for key, value in event.attributes.items():
                event_fields.append({"key": key, "type": "string", "value": str(value)})
            logs.append({
                "timestamp": int(event.timestamp.timestamp() * 1_000_000),
                "fields": event_fields,
            })
        
        # Convert references
        references = []
        if span.parent_span_id:
            references.append({
                "refType": "CHILD_OF",
                "traceID": span.trace_id.replace("-", "")[:32],
                "spanID": span.parent_span_id.replace("-", "")[:16],
            })
        
        return JaegerSpan(
            trace_id=span.trace_id,
            span_id=span.span_id,
            operation_name=span.name,
            references=references,
            start_time=int(span.start_time.timestamp() * 1_000_000),
            duration=int(span.duration_ms * 1000) if span.duration_ms else 0,
            tags=tags,
            logs=logs,
        )
    
    def export_span(self, span: Span) -> None:
        """Export a span to Jaeger."""
        jaeger_span = self._convert_span(span)
        
        if span.trace_id not in self._traces:
            self._traces[span.trace_id] = []
        self._traces[span.trace_id].append(jaeger_span)
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export multiple spans."""
        for span in spans:
            self.export_span(span)
    
    def get_traces(self) -> Dict[str, Any]:
        """Get all traces in Jaeger format."""
        data = []
        for trace_id, spans in self._traces.items():
            data.append({
                "traceID": trace_id.replace("-", "")[:32],
                "spans": [s.to_dict() for s in spans],
                "processes": {
                    "p1": {
                        "serviceName": self.service_name,
                        "tags": [],
                    }
                },
            })
        return {"data": data}
    
    def flush(self) -> None:
        """Flush traces to file or collector."""
        if self.output_file:
            with open(self.output_file, 'w') as f:
                json.dump(self.get_traces(), f, indent=2)
        
        if self.collector_endpoint:
            # In production, would use requests or httpx
            # For now, just log that we would send
            pass
    
    def clear(self) -> None:
        """Clear stored traces."""
        self._traces.clear()
