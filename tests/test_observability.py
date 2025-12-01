"""Tests for the observability module."""

import pytest
import time
from datetime import datetime
from io import StringIO

from agentic_chain.observability.tracer import (
    Tracer,
    Span,
    SpanStatus,
    SpanKind,
    TracerConfig,
    trace,
)
from agentic_chain.observability.metrics import (
    MetricsCollector,
    MetricType,
    Metric,
    Timer,
    timed,
)
from agentic_chain.observability.logging import (
    StructuredLogger,
    LogLevel,
    LogRecord,
)
from agentic_chain.observability.context import (
    TraceContext,
    ContextManager,
    ContextScope,
)
from agentic_chain.observability.exporters import (
    ConsoleExporter,
    JSONExporter,
    JaegerExporter,
    PrometheusExporter,
)
from agentic_chain.observability.dashboard import (
    AgentStep,
    ExecutionTimeline,
    ObservabilityData,
)


class TestTracer:
    """Tests for the Tracer class."""
    
    def test_init_default(self):
        """Test default tracer initialization."""
        tracer = Tracer()
        assert tracer.config.enabled is True
        assert tracer.config.service_name == "agentic-chain"
    
    def test_init_with_config(self):
        """Test tracer with custom config."""
        config = TracerConfig(service_name="test-service", enabled=False)
        tracer = Tracer(config)
        assert tracer.config.service_name == "test-service"
        assert tracer.config.enabled is False
    
    def test_start_span(self):
        """Test starting a span."""
        tracer = Tracer()
        span = tracer.start_span("test_operation")
        
        assert span.name == "test_operation"
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.status == SpanStatus.UNSET
    
    def test_span_context_manager(self):
        """Test span as context manager."""
        tracer = Tracer()
        
        with tracer.start_span("test_operation") as span:
            span.set_attribute("key", "value")
            assert not span.is_ended
        
        assert span.is_ended
        assert span.status == SpanStatus.OK
    
    def test_span_with_exception(self):
        """Test span records exception."""
        tracer = Tracer()
        
        with pytest.raises(ValueError):
            with tracer.start_span("test_operation") as span:
                raise ValueError("test error")
        
        assert span.status == SpanStatus.ERROR
        assert "test error" in span.status_message
        assert len(span.events) == 1
        assert span.events[0].name == "exception"
    
    def test_span_duration(self):
        """Test span duration calculation."""
        tracer = Tracer()
        
        with tracer.start_span("test_operation") as span:
            time.sleep(0.01)
        
        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms
    
    def test_span_parent_relationship(self):
        """Test parent-child span relationship."""
        tracer = Tracer()
        
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id
    
    def test_tracer_records_spans(self):
        """Test tracer records completed spans."""
        tracer = Tracer()
        
        with tracer.start_span("operation1") as span1:
            pass
        with tracer.start_span("operation2") as span2:
            pass
        
        assert len(tracer.spans) == 2
    
    def test_tracer_statistics(self):
        """Test tracer statistics."""
        tracer = Tracer()
        
        with tracer.start_span("success"):
            pass
        
        try:
            with tracer.start_span("error"):
                raise ValueError("test")
        except ValueError:
            pass
        
        stats = tracer.get_statistics()
        assert stats["total_spans"] == 2
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 0.5
    
    def test_trace_decorator(self):
        """Test trace decorator."""
        tracer = Tracer()
        
        @trace(tracer, "my_function")
        def test_function():
            return "result"
        
        result = test_function()
        assert result == "result"
        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "my_function"


class TestSpan:
    """Tests for the Span class."""
    
    def test_set_attribute(self):
        """Test setting span attribute."""
        span = Span(name="test")
        span.set_attribute("key", "value")
        
        assert span.attributes["key"] == "value"
    
    def test_set_attributes(self):
        """Test setting multiple attributes."""
        span = Span(name="test")
        span.set_attributes({"key1": "value1", "key2": "value2"})
        
        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
    
    def test_add_event(self):
        """Test adding event to span."""
        span = Span(name="test")
        span.add_event("my_event", {"detail": "info"})
        
        assert len(span.events) == 1
        assert span.events[0].name == "my_event"
        assert span.events[0].attributes["detail"] == "info"
    
    def test_to_dict(self):
        """Test span to dict conversion."""
        span = Span(name="test", kind=SpanKind.SERVER)
        span.set_attribute("key", "value")
        span.end()
        
        data = span.to_dict()
        assert data["name"] == "test"
        assert data["kind"] == "server"
        assert data["attributes"]["key"] == "value"
        assert data["duration_ms"] is not None


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""
    
    def test_init(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        assert collector.prefix == "agentic_chain"
    
    def test_record_execution_time(self):
        """Test recording execution time."""
        collector = MetricsCollector()
        collector.record_execution_time("TestAgent", 0.5, success=True)
        
        metric = collector.get_metric("agent_execution_duration_seconds")
        assert metric is not None
        assert len(metric.values) == 1
        assert metric.values[0].value == 0.5
    
    def test_record_execution_error(self):
        """Test recording execution error."""
        collector = MetricsCollector()
        collector.record_execution_time("TestAgent", 0.5, success=False)
        
        errors = collector.get_metric("agent_errors_total")
        assert errors is not None
        assert errors.get_sum() == 1
    
    def test_record_llm_usage(self):
        """Test recording LLM usage."""
        collector = MetricsCollector()
        collector.record_llm_usage(
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration_seconds=1.5,
            cost=0.01,
        )
        
        tokens = collector.get_metric("llm_tokens_total")
        assert tokens is not None
        assert tokens.get_sum() == 150
    
    def test_increment_counter(self):
        """Test incrementing counter."""
        collector = MetricsCollector()
        collector.increment_counter("custom_counter")
        collector.increment_counter("custom_counter")
        
        metric = collector.get_metric("custom_counter")
        assert metric.get_sum() == 2
    
    def test_set_gauge(self):
        """Test setting gauge."""
        collector = MetricsCollector()
        collector.set_gauge("memory_usage", 1024)
        collector.set_gauge("memory_usage", 2048)
        
        metric = collector.get_metric("memory_usage")
        assert metric.get_current_value() == 2048
    
    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()
        collector.record_execution_time("Agent1", 0.5)
        collector.record_execution_time("Agent2", 1.0)
        
        summary = collector.get_summary()
        assert summary["total_executions"] == 2
        assert summary["avg_execution_time_seconds"] == 0.75
    
    def test_to_prometheus(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        collector.increment_counter("test_counter")
        
        output = collector.to_prometheus()
        assert "agentic_chain_test_counter" in output


class TestMetric:
    """Tests for the Metric class."""
    
    def test_counter(self):
        """Test counter metric."""
        metric = Metric(name="test", type=MetricType.COUNTER)
        metric.record(1)
        metric.record(2)
        metric.record(3)
        
        assert metric.get_sum() == 6
    
    def test_gauge(self):
        """Test gauge metric."""
        metric = Metric(name="test", type=MetricType.GAUGE)
        metric.record(10)
        metric.record(20)
        metric.record(15)
        
        assert metric.get_current_value() == 15
    
    def test_histogram(self):
        """Test histogram metric."""
        metric = Metric(name="test", type=MetricType.HISTOGRAM)
        for i in range(100):
            metric.record(i / 100)  # 0.0 to 0.99
        
        buckets = metric.get_histogram_buckets()
        assert "le_0.5" in buckets
        assert buckets["le_inf"] == 100
    
    def test_percentile(self):
        """Test percentile calculation."""
        metric = Metric(name="test", type=MetricType.SUMMARY)
        for i in range(100):
            metric.record(i)
        
        p50 = metric.get_percentile(50)
        p95 = metric.get_percentile(95)
        
        assert 45 <= p50 <= 55
        assert 90 <= p95 <= 99


class TestTimer:
    """Tests for the Timer class."""
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        collector = MetricsCollector()
        
        with Timer(collector, "test_duration") as timer:
            time.sleep(0.01)
        
        assert timer.duration >= 0.01
        metric = collector.get_metric("test_duration")
        assert len(metric.values) == 1
    
    def test_timed_decorator(self):
        """Test timed decorator."""
        collector = MetricsCollector()
        
        @timed(collector, "func_duration")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"
        
        metric = collector.get_metric("func_duration")
        assert len(metric.values) == 1
        assert metric.values[0].value >= 0.01


class TestStructuredLogger:
    """Tests for the StructuredLogger class."""
    
    def test_init(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_logger_init")
        assert logger.name == "test_logger_init"
        assert logger.level == LogLevel.INFO
    
    def test_log_levels(self):
        """Test log level conversion."""
        assert LogLevel.DEBUG.to_python_level() == 10
        assert LogLevel.INFO.to_python_level() == 20
        assert LogLevel.WARNING.to_python_level() == 30
        assert LogLevel.ERROR.to_python_level() == 40
        assert LogLevel.CRITICAL.to_python_level() == 50
    
    def test_log_record_to_dict(self):
        """Test LogRecord to dict conversion."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
            trace_id="abc123",
            attributes={"key": "value"},
        )
        
        data = record.to_dict()
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert data["trace_id"] == "abc123"
        assert data["attributes"]["key"] == "value"
    
    def test_log_record_to_text(self):
        """Test LogRecord to text conversion."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
            trace_id="abc12345678901234",
        )
        
        text = record.to_text()
        assert "INFO" in text
        assert "Test message" in text
        assert "abc12345" in text  # trace_id truncated
    
    def test_with_context(self):
        """Test context logger."""
        logger = StructuredLogger("test_with_ctx")
        ctx_logger = logger.with_context(agent="TestAgent")
        
        # Context logger should merge attributes
        merged = ctx_logger._merge_attrs({"extra": "value"})
        assert merged["agent"] == "TestAgent"
        assert merged["extra"] == "value"
    
    def test_get_child(self):
        """Test getting child logger."""
        logger = StructuredLogger("parent")
        child = logger.get_child("child")
        
        assert child.name == "parent.child"


class TestTraceContext:
    """Tests for TraceContext."""
    
    def test_create_context(self):
        """Test creating trace context."""
        context = TraceContext()
        assert context.trace_id is not None
        assert context.span_id is not None
    
    def test_create_child(self):
        """Test creating child context."""
        parent = TraceContext()
        child = parent.create_child()
        
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.span_id != parent.span_id
    
    def test_set_attribute(self):
        """Test setting context attribute."""
        context = TraceContext()
        context.set_attribute("key", "value")
        
        assert context.attributes["key"] == "value"
    
    def test_to_dict(self):
        """Test context to dict conversion."""
        context = TraceContext()
        context.set_attribute("key", "value")
        
        data = context.to_dict()
        assert "trace_id" in data
        assert "span_id" in data
        assert data["attributes"]["key"] == "value"


class TestContextManager:
    """Tests for ContextManager."""
    
    def test_set_and_get_current(self):
        """Test setting and getting current context."""
        context = TraceContext()
        ContextManager.set_current(context)
        
        current = ContextManager.get_current()
        assert current.trace_id == context.trace_id
    
    def test_create_child_context(self):
        """Test creating child context."""
        parent = TraceContext()
        ContextManager.set_current(parent)
        
        child = ContextManager.create_child_context()
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
    
    def test_context_scope(self):
        """Test context scope."""
        original = TraceContext()
        ContextManager.set_current(original)
        
        new_context = TraceContext()
        with ContextScope(new_context):
            current = ContextManager.get_current()
            assert current.trace_id == new_context.trace_id
        
        # Should restore original
        restored = ContextManager.get_current()
        assert restored.trace_id == original.trace_id


class TestExporters:
    """Tests for exporters."""
    
    def test_console_exporter(self):
        """Test console exporter."""
        output = StringIO()
        exporter = ConsoleExporter(output=output, colored=False)
        
        span = Span(name="test_operation")
        span.set_attribute("key", "value")
        span.end()
        
        exporter.export_span(span)
        
        result = output.getvalue()
        assert "test_operation" in result
        assert "key" in result
    
    def test_json_exporter(self):
        """Test JSON exporter."""
        output = StringIO()
        exporter = JSONExporter(output=output)
        
        span = Span(name="test_operation")
        span.end()
        
        exporter.export_span(span)
        
        result = output.getvalue()
        assert '"name": "test_operation"' in result
    
    def test_jaeger_exporter(self):
        """Test Jaeger exporter."""
        exporter = JaegerExporter(service_name="test-service")
        
        span = Span(name="test_operation")
        span.set_attribute("key", "value")
        span.end()
        
        exporter.export_span(span)
        
        traces = exporter.get_traces()
        assert len(traces["data"]) == 1
        assert traces["data"][0]["spans"][0]["operationName"] == "test_operation"
    
    def test_prometheus_exporter(self):
        """Test Prometheus exporter."""
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector=collector)
        
        span = Span(name="test_operation")
        span.end()
        
        exporter.export_span(span)
        
        metrics = exporter.get_metrics()
        assert "spans_total" in metrics


class TestDashboard:
    """Tests for dashboard data structures."""
    
    def test_agent_step_from_span(self):
        """Test creating AgentStep from Span."""
        span = Span(name="agent.TestAgent")
        span.set_attribute("agent.name", "TestAgent")
        span.set_status(SpanStatus.OK)
        span.end()
        
        step = AgentStep.from_span(span)
        
        assert step.name == "agent.TestAgent"
        assert step.agent_name == "TestAgent"
        assert step.status == "success"
    
    def test_execution_timeline(self):
        """Test ExecutionTimeline."""
        timeline = ExecutionTimeline(trace_id="test-trace-123")
        
        step1 = AgentStep(name="step1", agent_name="Agent1", status="success")
        step2 = AgentStep(name="step2", agent_name="Agent2", status="success")
        
        timeline.add_step(step1)
        timeline.add_step(step2)
        timeline.complete(success=True)
        
        assert timeline.step_count == 2
        assert timeline.success_count == 2
        assert timeline.error_count == 0
        assert timeline.status == "success"
    
    def test_execution_timeline_from_spans(self):
        """Test creating timeline from spans."""
        spans = []
        
        span1 = Span(name="step1")
        span1.set_status(SpanStatus.OK)
        span1.end()
        spans.append(span1)
        
        span2 = Span(name="step2")
        span2.set_status(SpanStatus.OK)
        span2.end()
        spans.append(span2)
        
        timeline = ExecutionTimeline.from_spans(spans, "test-trace")
        
        assert timeline.step_count == 2
        assert timeline.status == "success"
    
    def test_observability_data(self):
        """Test ObservabilityData."""
        collector = MetricsCollector()
        collector.record_execution_time("Agent1", 0.5)
        collector.record_execution_time("Agent2", 1.0)
        
        timeline = ExecutionTimeline(trace_id="test")
        timeline.add_step(AgentStep(name="step1", agent_name="Agent1", status="success"))
        timeline.complete()
        
        data = ObservabilityData.from_collector(collector, [timeline])
        
        assert data.total_executions == 2
        assert data.success_rate > 0
        assert len(data.recent_timelines) == 1
    
    def test_observability_data_to_dict(self):
        """Test ObservabilityData to dict."""
        data = ObservabilityData(
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
        )
        
        result = data.to_dict()
        
        assert result["summary"]["total_executions"] == 10
        assert result["summary"]["success_rate"] == 80.0


class TestIntegration:
    """Integration tests for observability with AgenticChain."""
    
    def test_chain_with_tracing(self, tmp_path):
        """Test AgenticChain with tracing enabled."""
        from agentic_chain import AgenticChain
        
        # Create minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path), enable_tracing=True)
        result = chain.solve_issue({
            "title": "Test issue",
            "body": "Test body",
            "labels": []
        })
        
        # Check tracing data
        assert "trace_id" in result
        assert len(chain.tracer.spans) > 0
        
        # Check timeline
        timeline = chain.get_execution_timeline()
        assert timeline is not None
        assert timeline.step_count == 4  # 4 default agents
    
    def test_chain_metrics(self, tmp_path):
        """Test AgenticChain metrics collection."""
        from agentic_chain import AgenticChain
        
        # Create minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path))
        chain.solve_issue({
            "title": "Test issue",
            "body": "Test body",
            "labels": []
        })
        
        summary = chain.get_metrics_summary()
        assert summary["total_executions"] == 4  # 4 agents
        assert summary["total_errors"] == 0
    
    def test_chain_observability_data(self, tmp_path):
        """Test getting observability data from chain."""
        from agentic_chain import AgenticChain
        
        # Create minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(project_path=str(tmp_path))
        chain.solve_issue({
            "title": "Test issue",
            "body": "Test body",
            "labels": []
        })
        
        obs_data = chain.get_observability_data()
        assert obs_data.total_executions == 4
        assert len(obs_data.recent_timelines) == 1
    
    def test_chain_with_tracing_disabled(self, tmp_path):
        """Test AgenticChain with tracing disabled."""
        from agentic_chain import AgenticChain
        
        # Create minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        
        chain = AgenticChain(
            project_path=str(tmp_path),
            enable_tracing=False,
        )
        result = chain.solve_issue({
            "title": "Test issue",
            "body": "Test body",
            "labels": []
        })
        
        # Should still work, just no trace data
        assert result is not None
