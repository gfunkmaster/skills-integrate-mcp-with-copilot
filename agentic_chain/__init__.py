"""
Agentic Chain - The fastest way to get AI-powered insights on GitHub issues.

Stop spending hours triaging issues. Get AI-powered insights in seconds.

Key Features:
- Instant issue classification (< 5 seconds)
- Priority scoring algorithm
- Similar issue detection
- Sentiment analysis for urgency detection
- Auto-labeling suggestions
- Comprehensive observability with tracing, metrics, and logging
- GitHub integration for automated issue processing
This package provides an agentic chain that can understand project context,
analyze issues, review code, and implement solutions in external projects.

It supports LLM integration for intelligent code generation using multiple
providers (OpenAI, Anthropic).
"""

from .orchestrator import AgenticChain
from .agents.project_analyzer import ProjectAnalyzer
from .agents.issue_analyzer import IssueAnalyzer
from .agents.code_reviewer import CodeReviewer
from .agents.solution_implementer import SolutionImplementer
from .agents.similar_issue_detector import SimilarIssueDetector
from .agents import AgentContext, LLMContext

# LLM integration
from .llm import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMUsage,
    LLMFactory,
    OpenAIProvider,
    AnthropicProvider,
)

# Observability
from .observability import (
    Tracer,
    Span,
    SpanStatus,
    SpanKind,
    TracerConfig,
    MetricsCollector,
    MetricType,
    Metric,
    StructuredLogger,
    LogLevel,
    TraceContext,
    ContextManager,
    Exporter,
    ConsoleExporter,
    JSONExporter,
    PrometheusExporter,
    JaegerExporter,
    ObservabilityData,
    ExecutionTimeline,
    AgentStep,
)

# GitHub integration
from .github_integration import (
    WebhookHandler,
    IssueProcessor,
    CommentFormatter,
    GitHubConfig,
)

__version__ = "0.1.0"
__all__ = [
    # Core chain
    "AgenticChain",
    # Agents
    "ProjectAnalyzer",
    "IssueAnalyzer",
    "CodeReviewer",
    "SolutionImplementer",
    "SimilarIssueDetector",
    # Context
    "AgentContext",
    "LLMContext",
    # LLM
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "LLMMessage",
    "LLMUsage",
    "LLMFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    # Observability
    "Tracer",
    "Span",
    "SpanStatus",
    "SpanKind",
    "TracerConfig",
    "MetricsCollector",
    "MetricType",
    "Metric",
    "StructuredLogger",
    "LogLevel",
    "TraceContext",
    "ContextManager",
    "Exporter",
    "ConsoleExporter",
    "JSONExporter",
    "PrometheusExporter",
    "JaegerExporter",
    "ObservabilityData",
    "ExecutionTimeline",
    "AgentStep",
    # GitHub Integration
    "WebhookHandler",
    "IssueProcessor",
    "CommentFormatter",
    "GitHubConfig",
]
