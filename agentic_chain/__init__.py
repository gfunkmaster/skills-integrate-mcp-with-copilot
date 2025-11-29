"""
Agentic Chain - An AI-powered issue solving framework.

This package provides an agentic chain that can understand project context,
analyze issues, review code, and implement solutions in external projects.
"""

from .orchestrator import AgenticChain
from .agents.project_analyzer import ProjectAnalyzer
from .agents.issue_analyzer import IssueAnalyzer
from .agents.code_reviewer import CodeReviewer
from .agents.solution_implementer import SolutionImplementer

__version__ = "0.1.0"
__all__ = [
    "AgenticChain",
    "ProjectAnalyzer",
    "IssueAnalyzer",
    "CodeReviewer",
    "SolutionImplementer",
]
