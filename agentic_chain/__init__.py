"""
Agentic Chain - An AI-powered issue solving framework.

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

__version__ = "0.1.0"
__all__ = [
    # Core chain
    "AgenticChain",
    # Agents
    "ProjectAnalyzer",
    "IssueAnalyzer",
    "CodeReviewer",
    "SolutionImplementer",
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
]
