"""
LLM Integration Module - Provides abstraction for multiple LLM providers.

This module provides a unified interface for interacting with different
LLM providers like OpenAI, Anthropic, and local models.
"""

from .base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMUsage,
)
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .factory import LLMFactory
from .prompts import (
    PROJECT_ANALYSIS_SYSTEM,
    PROJECT_ANALYSIS_PROMPT,
    ISSUE_CLASSIFICATION_SYSTEM,
    ISSUE_CLASSIFICATION_PROMPT,
    CODE_REVIEW_SYSTEM,
    CODE_REVIEW_PROMPT,
    CODE_REVIEW_SUMMARY_PROMPT,
    IMPLEMENTATION_PLAN_SYSTEM,
    IMPLEMENTATION_PLAN_PROMPT,
    format_project_analysis_prompt,
    format_issue_classification_prompt,
    format_code_review_prompt,
    format_implementation_plan_prompt,
)

__all__ = [
    # Core classes
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "LLMMessage",
    "LLMUsage",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "LLMFactory",
    # Prompt templates
    "PROJECT_ANALYSIS_SYSTEM",
    "PROJECT_ANALYSIS_PROMPT",
    "ISSUE_CLASSIFICATION_SYSTEM",
    "ISSUE_CLASSIFICATION_PROMPT",
    "CODE_REVIEW_SYSTEM",
    "CODE_REVIEW_PROMPT",
    "CODE_REVIEW_SUMMARY_PROMPT",
    "IMPLEMENTATION_PLAN_SYSTEM",
    "IMPLEMENTATION_PLAN_PROMPT",
    "format_project_analysis_prompt",
    "format_issue_classification_prompt",
    "format_code_review_prompt",
    "format_implementation_plan_prompt",
]
