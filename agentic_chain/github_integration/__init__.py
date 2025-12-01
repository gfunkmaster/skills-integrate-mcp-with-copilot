"""
GitHub Integration for Agentic Chain.

This module provides GitHub integration for automated issue processing,
including webhook handlers, comment posting, and label management.
"""

from .webhook_handler import WebhookHandler
from .issue_processor import IssueProcessor
from .comment_formatter import CommentFormatter
from .config import GitHubConfig

__all__ = [
    "WebhookHandler",
    "IssueProcessor",
    "CommentFormatter",
    "GitHubConfig",
]
