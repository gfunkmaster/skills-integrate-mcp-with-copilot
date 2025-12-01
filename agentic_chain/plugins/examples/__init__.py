"""
Example plugins demonstrating the plugin system.

These plugins serve as both examples and useful utilities:
- SecurityAuditor: Scans for security vulnerabilities
- PerformanceAnalyzer: Identifies performance bottlenecks
- DocumentationChecker: Ensures code is properly documented
"""

from .security_auditor import SecurityAuditor
from .performance_analyzer import PerformanceAnalyzer
from .documentation_checker import DocumentationChecker

__all__ = [
    "SecurityAuditor",
    "PerformanceAnalyzer",
    "DocumentationChecker",
]
