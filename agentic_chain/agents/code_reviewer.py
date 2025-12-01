"""
Code Reviewer Agent - Reviews code and suggests improvements.

This agent can optionally use LLM to provide intelligent code review
suggestions in addition to static analysis.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from . import BaseAgent, AgentContext

if TYPE_CHECKING:
    from ..llm import LLMProvider


logger = logging.getLogger(__name__)


class CodeReviewer(BaseAgent):
    """
    Reviews code in the project to identify issues, patterns,
    and areas that need improvement based on the issue analysis.
    
    When an LLM provider is configured, provides AI-powered code
    review suggestions in addition to static analysis.
    """
    
    def __init__(
        self,
        name: str = "CodeReviewer",
        llm_provider: Optional["LLMProvider"] = None,
        max_files_for_llm: int = 5,
        max_lines_per_file: int = 500,
    ):
        super().__init__(name)
        self._llm_provider = llm_provider
        self.max_files_for_llm = max_files_for_llm
        self.max_lines_per_file = max_lines_per_file
    
    @property
    def llm_provider(self) -> Optional["LLMProvider"]:
        """Get the LLM provider."""
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider: "LLMProvider"):
        """Set the LLM provider."""
        self._llm_provider = provider
        
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Review code based on issue analysis and project structure.
        """
        if not context.project_analysis:
            raise ValueError("Project analysis is required for code review")
            
        project_path = Path(context.project_path)
        
        # Static analysis (always performed)
        review = {
            "relevant_files": self._find_relevant_files(context),
            "code_quality": self._analyze_code_quality(context),
            "potential_issues": self._identify_potential_issues(context),
            "suggestions": self._generate_suggestions(context),
            "file_analyses": {},
        }
        
        # Analyze each relevant file
        for file_path in review["relevant_files"]:
            full_path = project_path / file_path
            if full_path.exists():
                review["file_analyses"][file_path] = self._analyze_file(full_path)
        
        # LLM-powered code review (if provider is available)
        if self._llm_provider:
            try:
                llm_review = self._perform_llm_review(
                    review["relevant_files"],
                    project_path,
                    context,
                )
                if llm_review:
                    review["llm_insights"] = llm_review
                    review["llm_enhanced"] = True
            except Exception as e:
                logger.warning(f"LLM code review failed: {e}")
                review["llm_enhanced"] = False
        else:
            review["llm_enhanced"] = False
                
        context.code_review = review
        return context
    
    def _perform_llm_review(
        self,
        relevant_files: list,
        project_path: Path,
        context: AgentContext,
    ) -> Optional[dict]:
        """Perform LLM-powered code review."""
        if not self._llm_provider:
            return None
        
        from ..llm import LLMMessage, format_code_review_prompt
        from ..llm.base import MessageRole
        from ..llm.prompts import CODE_REVIEW_SYSTEM
        
        # Get issue summary
        issue_summary = "No issue context provided"
        if context.issue_analysis:
            title = context.issue_analysis.get("title", "")
            issue_type = context.issue_analysis.get("issue_type", "unknown")
            requirements = context.issue_analysis.get("requirements", [])[:3]
            issue_summary = f"Type: {issue_type}\nTitle: {title}"
            if requirements:
                issue_summary += f"\nRequirements:\n" + "\n".join(f"- {r}" for r in requirements)
        
        # Review the most relevant files
        file_reviews = []
        files_to_review = relevant_files[:self.max_files_for_llm]
        
        for file_path in files_to_review:
            full_path = project_path / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # Limit lines to prevent token overflow
                    if len(lines) > self.max_lines_per_file:
                        code = "".join(lines[:self.max_lines_per_file])
                        code += f"\n# ... (truncated, {len(lines) - self.max_lines_per_file} more lines)"
                    else:
                        code = "".join(lines)
            except (IOError, OSError):
                continue
            
            # Detect language from extension
            ext = full_path.suffix.lower()
            language_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".go": "go", ".java": "java", ".rb": "ruby",
                ".rs": "rust", ".cpp": "cpp", ".c": "c",
            }
            language = language_map.get(ext, "")
            
            # Build prompt
            prompt = format_code_review_prompt(
                code=code,
                file_path=file_path,
                issue_summary=issue_summary,
                language=language,
            )
            
            messages = [
                LLMMessage(role=MessageRole.SYSTEM, content=CODE_REVIEW_SYSTEM),
                LLMMessage(role=MessageRole.USER, content=prompt),
            ]
            
            response = self._llm_provider.complete(messages)
            
            # Track usage
            if response.usage:
                context.llm_context.add_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost=response.usage.estimated_cost,
                )
                context.llm_context.provider = self._llm_provider.config.provider
                context.llm_context.model = response.model
            
            context.llm_context.responses.append({
                "type": "code_review",
                "file": file_path,
                "content": response.content,
                "model": response.model,
            })
            
            # Parse review from response
            parsed_review = self._parse_review_response(response.content)
            
            file_reviews.append({
                "file": file_path,
                "raw_content": response.content,
                "parsed": parsed_review,
                "model": response.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            })
        
        return {
            "file_reviews": file_reviews,
            "files_reviewed": len(file_reviews),
        }
    
    def _parse_review_response(self, content: str) -> Optional[dict]:
        """Parse JSON from LLM review response."""
        try:
            # Try to find JSON block in response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Try parsing the entire response as JSON
            return json.loads(content)
        except json.JSONDecodeError:
            return None
                
        context.code_review = review
        return context
    
    def _find_relevant_files(self, context: AgentContext) -> list:
        """Find files relevant to the issue."""
        relevant_files = []
        project_path = Path(context.project_path)
        
        # Start with files mentioned in issue
        if context.issue_analysis:
            mentioned_files = context.issue_analysis.get("affected_files", [])
            for file_ref in mentioned_files:
                # Search for the file in project
                for root, dirs, files in os.walk(project_path):
                    dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__"}]
                    for f in files:
                        if f == file_ref or file_ref in os.path.join(root, f):
                            rel_path = os.path.relpath(os.path.join(root, f), project_path)
                            relevant_files.append(rel_path)
                            
        # Find files matching keywords
        if context.issue_analysis:
            keywords = context.issue_analysis.get("keywords", [])
            for root, dirs, files in os.walk(project_path):
                dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__"}]
                for f in files:
                    # Check if filename matches any keyword
                    for kw in keywords:
                        if kw in f.lower():
                            rel_path = os.path.relpath(os.path.join(root, f), project_path)
                            relevant_files.append(rel_path)
                            break
                            
        # Include main source files if no relevant files found
        if not relevant_files:
            for root, dirs, files in os.walk(project_path):
                dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__"}]
                for f in files:
                    if f.endswith(('.py', '.js', '.ts', '.go', '.java', '.rb')):
                        rel_path = os.path.relpath(os.path.join(root, f), project_path)
                        relevant_files.append(rel_path)
                        if len(relevant_files) >= 10:
                            break
                if len(relevant_files) >= 10:
                    break
                    
        return list(set(relevant_files))[:20]  # Limit to 20 files
    
    def _analyze_code_quality(self, context: AgentContext) -> dict:
        """Analyze overall code quality metrics."""
        project_path = Path(context.project_path)
        
        metrics = {
            "has_tests": False,
            "has_documentation": False,
            "has_type_hints": False,
            "has_linting_config": False,
            "complexity_warnings": [],
        }
        
        # Check for tests
        test_dirs = ["tests", "test", "spec", "__tests__"]
        for test_dir in test_dirs:
            if (project_path / test_dir).exists():
                metrics["has_tests"] = True
                break
                
        # Check for documentation
        doc_files = ["README.md", "docs", "documentation"]
        for doc in doc_files:
            if (project_path / doc).exists():
                metrics["has_documentation"] = True
                break
                
        # Check for linting config
        lint_configs = [".eslintrc", ".pylintrc", ".flake8", "pyproject.toml", "setup.cfg"]
        for lint in lint_configs:
            if (project_path / lint).exists():
                metrics["has_linting_config"] = True
                break
                
        # Simple type hint detection for Python
        if context.project_analysis:
            if "Python" in context.project_analysis.get("languages", {}):
                for root, dirs, files in os.walk(project_path):
                    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
                    for f in files:
                        if f.endswith('.py'):
                            try:
                                with open(os.path.join(root, f), 'r', encoding='utf-8') as fp:
                                    content = fp.read()
                                    if re.search(r'def\s+\w+\([^)]*:\s*\w+', content):
                                        metrics["has_type_hints"] = True
                                        break
                            except (IOError, UnicodeDecodeError):
                                pass
                    if metrics["has_type_hints"]:
                        break
                        
        return metrics
    
    def _identify_potential_issues(self, context: AgentContext) -> list:
        """Identify potential issues in the code."""
        issues = []
        project_path = Path(context.project_path)
        
        patterns_to_check = {
            "TODO": "Unfinished work",
            "FIXME": "Known bugs or issues",
            "HACK": "Workarounds that should be improved",
            "XXX": "Problematic or dangerous code",
            "BUG": "Known bugs",
        }
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__"}]
            for f in files:
                if f.endswith(('.py', '.js', '.ts', '.go', '.java', '.rb')):
                    file_path = os.path.join(root, f)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                            for line_num, line in enumerate(fp, 1):
                                for pattern, description in patterns_to_check.items():
                                    if pattern in line:
                                        rel_path = os.path.relpath(file_path, project_path)
                                        issues.append({
                                            "file": rel_path,
                                            "line": line_num,
                                            "type": pattern,
                                            "description": description,
                                            "content": line.strip()[:100],
                                        })
                    except IOError:
                        pass
                        
        return issues[:50]  # Limit to 50 issues
    
    def _generate_suggestions(self, context: AgentContext) -> list:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        if context.code_review and context.code_review.get("code_quality"):
            quality = context.code_review["code_quality"]
            
            if not quality["has_tests"]:
                suggestions.append({
                    "type": "testing",
                    "priority": "high",
                    "suggestion": "Add unit tests to improve code reliability",
                })
                
            if not quality["has_documentation"]:
                suggestions.append({
                    "type": "documentation",
                    "priority": "medium",
                    "suggestion": "Add documentation or improve README",
                })
                
            if not quality["has_type_hints"]:
                suggestions.append({
                    "type": "type_safety",
                    "priority": "low",
                    "suggestion": "Consider adding type hints for better code clarity",
                })
                
        # Issue-specific suggestions
        if context.issue_analysis:
            issue_type = context.issue_analysis.get("issue_type")
            
            if issue_type == "bug":
                suggestions.append({
                    "type": "bug_fix",
                    "priority": "high",
                    "suggestion": "Add regression tests for the bug fix",
                })
                
            if issue_type == "feature":
                suggestions.append({
                    "type": "feature",
                    "priority": "medium",
                    "suggestion": "Ensure feature is well-documented and tested",
                })
                
        return suggestions
    
    def _analyze_file(self, file_path: Path) -> dict:
        """Analyze a single file."""
        analysis = {
            "lines": 0,
            "blank_lines": 0,
            "comment_lines": 0,
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_warnings": [],
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                analysis["lines"] = len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        analysis["blank_lines"] += 1
                    elif stripped.startswith('#') or stripped.startswith('//'):
                        analysis["comment_lines"] += 1
                        
                # Extract functions (Python style)
                analysis["functions"] = re.findall(r'def\s+(\w+)\s*\(', content)
                
                # Extract classes
                analysis["classes"] = re.findall(r'class\s+(\w+)', content)
                
                # Extract imports
                analysis["imports"] = re.findall(r'(?:import|from)\s+(\S+)', content)
                
                # Simple complexity check
                for i, line in enumerate(lines, 1):
                    indent = len(line) - len(line.lstrip())
                    if indent > 40:  # Very deep nesting
                        analysis["complexity_warnings"].append(
                            f"Line {i}: Deep nesting detected"
                        )
                        
        except (IOError, UnicodeDecodeError):
            pass
            
        return analysis
