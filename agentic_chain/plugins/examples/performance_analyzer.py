"""
Performance Analyzer Plugin - Identifies performance bottlenecks.

This plugin demonstrates how to create a custom agent that analyzes
code for potential performance issues.
"""

import logging
import os
import re
from pathlib import Path
from typing import List

from ...agents import BaseAgent, AgentContext


logger = logging.getLogger(__name__)


class PerformanceAnalyzer(BaseAgent):
    """
    Analyzes project code for potential performance bottlenecks.
    
    This plugin checks for:
    - Inefficient loop patterns
    - N+1 query patterns
    - Large file operations without streaming
    - Synchronous I/O in async code
    - Memory-inefficient patterns
    
    Example:
        from agentic_chain.plugins.examples import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        context = analyzer.execute(context)
        
        # Results are stored in context.plugin_results["performance_analysis"]
    """
    
    # Performance anti-patterns
    # Note: These are heuristic patterns that may have false positives
    LOOP_PATTERNS = [
        (r'for\s+\w+\s+in\s+[^:]+:\s*\n\s+for\s+\w+\s+in\s+', "Nested loops detected - O(n²) complexity"),
        (r'\.append\([^)]+\)\s*$', "List append in loop - consider list comprehension"),
        (r'while\s+True:', "Infinite loop pattern - ensure proper exit condition"),
    ]
    
    QUERY_PATTERNS = [
        (r'for\s+\w+\s+in\s+\w+\.all\(\):', "N+1 query pattern - consider select_related/prefetch_related"),
        (r'\.filter\([^)]*\)\s*\n\s*\.filter\(', "Chained filters - consider combining into single query"),
        (r'for\s+\w+\s+in\s+.*\.objects\.', "Query in loop - potential N+1 issue"),
    ]
    
    IO_PATTERNS = [
        (r'open\([^)]+\)\.read\(\)', "Reading entire file into memory"),
        (r'\.read\(\)\s*$', "Reading entire content at once - consider chunked reading"),
        (r'json\.loads?\([^)]*\.read\(\)', "Loading entire JSON file into memory"),
        (r'requests\.get\([^)]+\)(?!.*stream)', "HTTP request without streaming"),
    ]
    
    MEMORY_PATTERNS = [
        (r'list\(range\(\d{6,}\)\)', "Large range converted to list"),
        (r'\*\s*\d{6,}', "Large list multiplication"),
        (r'\.readlines\(\)', "readlines() loads all lines into memory"),
        (r'\+\s*=\s*["\']', "String concatenation in loop - use join()"),
    ]
    
    ASYNC_PATTERNS = [
        (r'async\s+def\s+\w+.*:\s*\n(?:.*\n)*?.*time\.sleep\(', "Blocking sleep in async function"),
        (r'async\s+def\s+\w+.*:\s*\n(?:.*\n)*?.*requests\.', "Blocking HTTP in async function"),
        (r'async\s+def\s+\w+.*:\s*\n(?:.*\n)*?.*open\(', "Blocking file I/O in async function"),
    ]
    
    def __init__(self, name: str = "PerformanceAnalyzer"):
        super().__init__(name)
    
    @property
    def dependencies(self) -> List[str]:
        """PerformanceAnalyzer benefits from project analysis context."""
        return ["ProjectAnalyzer"]
    
    @property
    def description(self) -> str:
        return "Identifies potential performance bottlenecks in code"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute performance analysis on the project.
        
        Args:
            context: The agent context with project information.
            
        Returns:
            Updated context with performance findings in plugin_results.
        """
        project_path = Path(context.project_path)
        
        findings = []
        
        # Scan Python files
        findings.extend(self._scan_files(project_path, ['.py']))
        
        # Scan JavaScript/TypeScript files
        findings.extend(self._scan_files(project_path, ['.js', '.ts']))
        
        # Calculate complexity metrics
        complexity_metrics = self._analyze_complexity(project_path)
        
        # Categorize findings
        findings_by_category = {}
        for finding in findings:
            category = finding.get("category", "other")
            if category not in findings_by_category:
                findings_by_category[category] = []
            findings_by_category[category].append(finding)
        
        analysis_result = {
            "total_findings": len(findings),
            "findings": findings,
            "findings_by_category": findings_by_category,
            "complexity_metrics": complexity_metrics,
            "hotspots": self._identify_hotspots(findings),
            "recommendations": self._generate_recommendations(findings),
        }
        
        context.plugin_results["performance_analysis"] = analysis_result
        logger.info(f"Performance analysis complete: {len(findings)} potential issues found")
        
        return context
    
    def _scan_files(self, directory: Path, extensions: List[str]) -> List[dict]:
        """Scan files for performance issues."""
        findings = []
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue
                
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(directory)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Check patterns
                        findings.extend(self._check_patterns(
                            str(rel_path), content, self.LOOP_PATTERNS, "loop", "medium"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), content, self.QUERY_PATTERNS, "query", "high"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), content, self.IO_PATTERNS, "io", "medium"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), content, self.MEMORY_PATTERNS, "memory", "high"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), content, self.ASYNC_PATTERNS, "async", "high"
                        ))
                        
                        # Check function complexity
                        findings.extend(self._check_function_complexity(str(rel_path), content))
                        
                except (IOError, OSError) as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        return findings
    
    def _check_patterns(
        self,
        file_path: str,
        content: str,
        patterns: List[tuple],
        category: str,
        impact: str,
    ) -> List[dict]:
        """Check content against patterns."""
        findings = []
        lines = content.split('\n')
        
        for pattern, description in patterns:
            # Check multi-line patterns against full content
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    "file": file_path,
                    "line": line_num,
                    "category": category,
                    "impact": impact,
                    "description": description,
                    "code_snippet": match.group(0)[:100].replace('\n', ' '),
                })
        
        return findings
    
    def _check_function_complexity(self, file_path: str, content: str) -> List[dict]:
        """Check for overly complex functions."""
        findings = []
        
        # Simple complexity check: count nested blocks
        lines = content.split('\n')
        current_function = None
        function_start = 0
        max_indent = 0
        
        for line_num, line in enumerate(lines, 1):
            # Detect function definition
            func_match = re.match(r'^(async\s+)?def\s+(\w+)\s*\(', line)
            if func_match:
                # Check previous function
                if current_function and max_indent > 6:
                    findings.append({
                        "file": file_path,
                        "line": function_start,
                        "category": "complexity",
                        "impact": "medium",
                        "description": f"Function '{current_function}' has deep nesting (depth: {max_indent // 4})",
                        "code_snippet": f"def {current_function}...",
                    })
                
                current_function = func_match.group(2)
                function_start = line_num
                max_indent = 0
            
            # Track indentation
            if line.strip() and current_function:
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        return findings
    
    def _analyze_complexity(self, directory: Path) -> dict:
        """Calculate overall complexity metrics."""
        metrics = {
            "total_lines": 0,
            "code_lines": 0,
            "function_count": 0,
            "class_count": 0,
            "average_function_length": 0,
        }
        
        function_lengths = []
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for filename in files:
                if not filename.endswith('.py'):
                    continue
                
                file_path = Path(root) / filename
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        metrics["total_lines"] += len(lines)
                        metrics["code_lines"] += sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))
                        
                        # Count functions and classes
                        metrics["function_count"] += len(re.findall(r'^(async\s+)?def\s+\w+', content, re.MULTILINE))
                        metrics["class_count"] += len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
                        
                except (IOError, OSError):
                    pass
        
        if metrics["function_count"] > 0:
            metrics["average_function_length"] = metrics["code_lines"] // metrics["function_count"]
        
        return metrics
    
    def _identify_hotspots(self, findings: List[dict]) -> List[dict]:
        """Identify files with the most issues."""
        file_counts = {}
        for finding in findings:
            file_path = finding.get("file", "unknown")
            if file_path not in file_counts:
                file_counts[file_path] = {"count": 0, "categories": set()}
            file_counts[file_path]["count"] += 1
            file_counts[file_path]["categories"].add(finding.get("category", "other"))
        
        hotspots = []
        for file_path, data in sorted(file_counts.items(), key=lambda x: -x[1]["count"]):
            if data["count"] >= 2:
                hotspots.append({
                    "file": file_path,
                    "issue_count": data["count"],
                    "categories": list(data["categories"]),
                })
        
        return hotspots[:10]
    
    def _generate_recommendations(self, findings: List[dict]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        categories = {f.get("category") for f in findings}
        
        if "loop" in categories:
            recommendations.append("Consider using list comprehensions or generators instead of explicit loops")
        if "query" in categories:
            recommendations.append("Optimize database queries - use select_related/prefetch_related to avoid N+1 queries")
        if "io" in categories:
            recommendations.append("Use streaming and chunked reading for large file operations")
        if "memory" in categories:
            recommendations.append("Use generators and iterators to reduce memory footprint")
        if "async" in categories:
            recommendations.append("Replace blocking calls with async equivalents in async functions")
        if "complexity" in categories:
            recommendations.append("Refactor complex functions - extract smaller, focused functions")
        
        if not recommendations:
            recommendations.append("No major performance issues detected - continue monitoring")
        
        return recommendations
