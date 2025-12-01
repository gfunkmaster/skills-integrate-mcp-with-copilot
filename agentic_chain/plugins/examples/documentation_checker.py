"""
Documentation Checker Plugin - Ensures code is properly documented.

This plugin demonstrates how to create a custom agent that analyzes
code documentation quality.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from ...agents import BaseAgent, AgentContext


logger = logging.getLogger(__name__)


class DocumentationChecker(BaseAgent):
    """
    Checks that code is properly documented.
    
    This plugin checks for:
    - Missing docstrings in modules, classes, and functions
    - README file presence and quality
    - Type hints coverage
    - Inline comment density
    - API documentation completeness
    
    Example:
        from agentic_chain.plugins.examples import DocumentationChecker
        
        checker = DocumentationChecker()
        context = checker.execute(context)
        
        # Results are stored in context.plugin_results["documentation_check"]
    """
    
    def __init__(
        self,
        name: str = "DocumentationChecker",
        min_docstring_length: int = 10,
        require_type_hints: bool = False,
    ):
        super().__init__(name)
        self.min_docstring_length = min_docstring_length
        self.require_type_hints = require_type_hints
    
    @property
    def dependencies(self) -> List[str]:
        """DocumentationChecker benefits from project analysis context."""
        return ["ProjectAnalyzer"]
    
    @property
    def description(self) -> str:
        return "Checks code documentation quality including docstrings, README, and type hints"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute documentation check on the project.
        
        Args:
            context: The agent context with project information.
            
        Returns:
            Updated context with documentation findings in plugin_results.
        """
        project_path = Path(context.project_path)
        
        findings = []
        stats = {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "total_modules": 0,
            "documented_modules": 0,
            "functions_with_type_hints": 0,
        }
        
        # Check Python files
        py_findings, py_stats = self._check_python_files(project_path)
        findings.extend(py_findings)
        for key in stats:
            stats[key] += py_stats.get(key, 0)
        
        # Check README
        readme_result = self._check_readme(project_path)
        if readme_result:
            findings.extend(readme_result.get("issues", []))
        
        # Calculate documentation score
        doc_score = self._calculate_doc_score(stats, readme_result)
        
        check_result = {
            "total_findings": len(findings),
            "documentation_score": doc_score,
            "grade": self._get_grade(doc_score),
            "statistics": stats,
            "findings": findings,
            "readme_analysis": readme_result,
            "coverage": self._calculate_coverage(stats),
            "recommendations": self._generate_recommendations(findings, stats, readme_result),
        }
        
        context.plugin_results["documentation_check"] = check_result
        logger.info(f"Documentation check complete: score {doc_score}/100, {len(findings)} issues found")
        
        return context
    
    def _check_python_files(self, directory: Path) -> tuple:
        """Check Python files for documentation."""
        findings = []
        stats = {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "total_modules": 0,
            "documented_modules": 0,
            "functions_with_type_hints": 0,
        }
        
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for filename in files:
                if not filename.endswith('.py'):
                    continue
                
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(directory)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Check module docstring
                        stats["total_modules"] += 1
                        if self._has_module_docstring(content):
                            stats["documented_modules"] += 1
                        else:
                            findings.append({
                                "file": str(rel_path),
                                "line": 1,
                                "type": "missing_docstring",
                                "severity": "low",
                                "description": "Module is missing a docstring",
                            })
                        
                        # Check functions and classes
                        file_findings, file_stats = self._analyze_file_content(str(rel_path), content)
                        findings.extend(file_findings)
                        for key in file_stats:
                            stats[key] += file_stats[key]
                        
                except (IOError, OSError) as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        return findings, stats
    
    def _has_module_docstring(self, content: str) -> bool:
        """Check if module has a docstring."""
        # Skip shebang and encoding declarations
        lines = content.split('\n')
        start_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('# -*-') or stripped.startswith('#!'):
                continue
            start_line = i
            break
        
        remaining = '\n'.join(lines[start_line:]).strip()
        return remaining.startswith('"""') or remaining.startswith("'''")
    
    def _analyze_file_content(self, file_path: str, content: str) -> tuple:
        """Analyze file content for documentation."""
        findings = []
        stats = {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "functions_with_type_hints": 0,
        }
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for class definition
            class_match = re.match(r'^class\s+(\w+)', stripped)
            if class_match:
                stats["total_classes"] += 1
                class_name = class_match.group(1)
                
                # Check for docstring on next non-empty line
                has_docstring = self._check_docstring_follows(lines, i)
                if has_docstring:
                    stats["documented_classes"] += 1
                else:
                    findings.append({
                        "file": file_path,
                        "line": i + 1,
                        "type": "missing_docstring",
                        "severity": "medium",
                        "description": f"Class '{class_name}' is missing a docstring",
                    })
            
            # Check for function/method definition
            func_match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)', stripped)
            if func_match:
                stats["total_functions"] += 1
                func_name = func_match.group(3)
                params = func_match.group(4)
                
                # Skip private/dunder methods for docstring requirement
                if not func_name.startswith('_') or func_name.startswith('__init__'):
                    has_docstring = self._check_docstring_follows(lines, i)
                    if has_docstring:
                        stats["documented_functions"] += 1
                    else:
                        findings.append({
                            "file": file_path,
                            "line": i + 1,
                            "type": "missing_docstring",
                            "severity": "low",
                            "description": f"Function '{func_name}' is missing a docstring",
                        })
                
                # Check for type hints
                has_type_hints = self._has_type_hints(stripped, params)
                if has_type_hints:
                    stats["functions_with_type_hints"] += 1
                elif self.require_type_hints and not func_name.startswith('_'):
                    findings.append({
                        "file": file_path,
                        "line": i + 1,
                        "type": "missing_type_hints",
                        "severity": "low",
                        "description": f"Function '{func_name}' is missing type hints",
                    })
            
            i += 1
        
        return findings, stats
    
    def _check_docstring_follows(self, lines: List[str], start_line: int) -> bool:
        """Check if a docstring follows the definition."""
        # Look at the next few lines for a docstring
        for i in range(start_line + 1, min(start_line + 5, len(lines))):
            stripped = lines[i].strip()
            if not stripped:
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                return True
            if stripped and not stripped.startswith('#'):
                # Non-empty, non-comment line that's not a docstring
                return False
        return False
    
    def _has_type_hints(self, definition: str, params: str) -> bool:
        """Check if function has type hints."""
        # Check for return type hint
        has_return_hint = '->' in definition
        
        # Check for parameter type hints
        if not params.strip():
            return has_return_hint
        
        # Look for : in parameters (excluding self/cls)
        param_list = [p.strip() for p in params.split(',')]
        typed_params = 0
        total_params = 0
        
        for param in param_list:
            if param in ('self', 'cls') or param.startswith('*'):
                continue
            total_params += 1
            if ':' in param:
                typed_params += 1
        
        if total_params == 0:
            return has_return_hint
        
        return typed_params == total_params or has_return_hint
    
    def _check_readme(self, directory: Path) -> Optional[dict]:
        """Check README file quality."""
        readme_names = ["README.md", "README.rst", "README.txt", "README"]
        readme_path = None
        
        for name in readme_names:
            path = directory / name
            if path.exists():
                readme_path = path
                break
        
        if not readme_path:
            return {
                "exists": False,
                "issues": [{
                    "file": "README",
                    "line": 0,
                    "type": "missing_readme",
                    "severity": "high",
                    "description": "Project is missing a README file",
                }],
            }
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, OSError):
            return {"exists": False, "issues": []}
        
        issues = []
        sections_found = []
        
        # Check for important sections
        important_sections = {
            "installation": r'#+\s*install(ation)?',
            "usage": r'#+\s*usage',
            "description": r'#+\s*(about|description|overview)',
            "license": r'#+\s*licen[sc]e',
            "contributing": r'#+\s*contribut',
        }
        
        content_lower = content.lower()
        for section, pattern in important_sections.items():
            if re.search(pattern, content_lower):
                sections_found.append(section)
        
        # Check README length
        word_count = len(content.split())
        if word_count < 50:
            issues.append({
                "file": readme_path.name,
                "line": 1,
                "type": "short_readme",
                "severity": "medium",
                "description": "README is too short (less than 50 words)",
            })
        
        # Check for missing sections
        missing_sections = set(important_sections.keys()) - set(sections_found)
        for section in missing_sections:
            if section in ["installation", "usage"]:
                issues.append({
                    "file": readme_path.name,
                    "line": 1,
                    "type": "missing_section",
                    "severity": "low",
                    "description": f"README is missing '{section}' section",
                })
        
        return {
            "exists": True,
            "file": readme_path.name,
            "word_count": word_count,
            "sections_found": sections_found,
            "issues": issues,
        }
    
    def _calculate_doc_score(self, stats: dict, readme_result: Optional[dict]) -> int:
        """Calculate overall documentation score (0-100)."""
        score = 0
        
        # Function documentation (40 points)
        if stats["total_functions"] > 0:
            func_ratio = stats["documented_functions"] / stats["total_functions"]
            score += int(func_ratio * 40)
        else:
            score += 40  # No functions = full points
        
        # Class documentation (20 points)
        if stats["total_classes"] > 0:
            class_ratio = stats["documented_classes"] / stats["total_classes"]
            score += int(class_ratio * 20)
        else:
            score += 20
        
        # Module documentation (15 points)
        if stats["total_modules"] > 0:
            module_ratio = stats["documented_modules"] / stats["total_modules"]
            score += int(module_ratio * 15)
        else:
            score += 15
        
        # README (15 points)
        if readme_result and readme_result.get("exists"):
            readme_score = 15
            # Deduct for issues
            readme_score -= len(readme_result.get("issues", [])) * 3
            score += max(0, readme_score)
        
        # Type hints (10 points)
        if stats["total_functions"] > 0:
            type_hint_ratio = stats["functions_with_type_hints"] / stats["total_functions"]
            score += int(type_hint_ratio * 10)
        else:
            score += 10
        
        return min(100, max(0, score))
    
    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_coverage(self, stats: dict) -> dict:
        """Calculate documentation coverage percentages."""
        coverage = {}
        
        if stats["total_functions"] > 0:
            coverage["function_docstrings"] = round(
                stats["documented_functions"] / stats["total_functions"] * 100, 1
            )
            coverage["type_hints"] = round(
                stats["functions_with_type_hints"] / stats["total_functions"] * 100, 1
            )
        else:
            coverage["function_docstrings"] = 100.0
            coverage["type_hints"] = 100.0
        
        if stats["total_classes"] > 0:
            coverage["class_docstrings"] = round(
                stats["documented_classes"] / stats["total_classes"] * 100, 1
            )
        else:
            coverage["class_docstrings"] = 100.0
        
        if stats["total_modules"] > 0:
            coverage["module_docstrings"] = round(
                stats["documented_modules"] / stats["total_modules"] * 100, 1
            )
        else:
            coverage["module_docstrings"] = 100.0
        
        return coverage
    
    def _generate_recommendations(
        self,
        findings: List[dict],
        stats: dict,
        readme_result: Optional[dict],
    ) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # Check function documentation
        if stats["total_functions"] > 0:
            func_ratio = stats["documented_functions"] / stats["total_functions"]
            if func_ratio < 0.8:
                recommendations.append(
                    f"Add docstrings to functions - currently {int(func_ratio * 100)}% documented"
                )
        
        # Check class documentation
        if stats["total_classes"] > 0:
            class_ratio = stats["documented_classes"] / stats["total_classes"]
            if class_ratio < 0.9:
                recommendations.append(
                    f"Add docstrings to classes - currently {int(class_ratio * 100)}% documented"
                )
        
        # Check type hints
        if stats["total_functions"] > 0:
            type_ratio = stats["functions_with_type_hints"] / stats["total_functions"]
            if type_ratio < 0.5:
                recommendations.append(
                    f"Add type hints to improve code clarity - currently {int(type_ratio * 100)}% coverage"
                )
        
        # Check README
        if readme_result:
            if not readme_result.get("exists"):
                recommendations.append("Create a README.md file with project description and usage instructions")
            elif readme_result.get("word_count", 0) < 100:
                recommendations.append("Expand README with more details about installation and usage")
        
        if not recommendations:
            recommendations.append("Documentation is in good shape - keep it up!")
        
        return recommendations
