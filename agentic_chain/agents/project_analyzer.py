"""
Project Analyzer Agent - Understands project structure, dependencies, and patterns.

This agent can optionally use LLM to provide deeper architectural insights.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from . import BaseAgent, AgentContext

if TYPE_CHECKING:
    from ..llm import LLMProvider


logger = logging.getLogger(__name__)


class ProjectAnalyzer(BaseAgent):
    """
    Analyzes a project to understand its structure, dependencies,
    coding patterns, and conventions.
    
    When an LLM provider is configured, provides AI-powered architectural
    insights in addition to static analysis.
    """
    
    def __init__(
        self,
        name: str = "ProjectAnalyzer",
        llm_provider: Optional["LLMProvider"] = None,
    ):
        super().__init__(name)
        self._llm_provider = llm_provider
    
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
        Analyze the project and populate context with analysis results.
        """
        if not self.validate_context(context):
            raise ValueError("Invalid context: project_path is required")
            
        project_path = Path(context.project_path)
        
        # Static analysis (always performed)
        analysis = {
            "structure": self._analyze_structure(project_path),
            "dependencies": self._analyze_dependencies(project_path),
            "languages": self._detect_languages(project_path),
            "patterns": self._detect_patterns(project_path),
            "readme": self._read_readme(project_path),
            "config_files": self._find_config_files(project_path),
        }
        
        # LLM-powered analysis (if provider is available)
        if self._llm_provider:
            try:
                llm_analysis = self._perform_llm_analysis(analysis, context)
                if llm_analysis:
                    analysis["llm_insights"] = llm_analysis
                    analysis["llm_enhanced"] = True
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
                analysis["llm_enhanced"] = False
        else:
            analysis["llm_enhanced"] = False
        
        context.project_analysis = analysis
        return context
    
    def _perform_llm_analysis(
        self,
        static_analysis: dict,
        context: AgentContext,
    ) -> Optional[dict]:
        """Perform LLM-powered project analysis."""
        if not self._llm_provider:
            return None
        
        from ..llm import LLMMessage, format_project_analysis_prompt
        from ..llm.base import MessageRole
        from ..llm.prompts import PROJECT_ANALYSIS_SYSTEM
        
        # Format the analysis data for the prompt
        structure = static_analysis.get("structure", {})
        directories = structure.get("directories", [])[:20]  # Limit to 20 dirs
        dir_structure = "\n".join(f"- {d}" for d in directories) or "No directories"
        
        # Format dependencies
        deps = static_analysis.get("dependencies", {})
        dep_lines = []
        for lang, dep_data in deps.items():
            if dep_data:
                if isinstance(dep_data, list):
                    dep_lines.append(f"**{lang}**: {', '.join(dep_data[:10])}")
                elif isinstance(dep_data, dict):
                    all_deps = list(dep_data.get("dependencies", {}).keys())[:10]
                    dep_lines.append(f"**{lang}**: {', '.join(all_deps)}")
        dependencies = "\n".join(dep_lines) or "No dependencies detected"
        
        # Format languages
        languages = static_analysis.get("languages", {})
        lang_str = ", ".join(f"{k}: {v} files" for k, v in languages.items()) or "None detected"
        
        # Format config files
        config_files = ", ".join(static_analysis.get("config_files", [])[:10]) or "None"
        
        # Get README
        readme = static_analysis.get("readme")
        
        # Build prompt
        prompt = format_project_analysis_prompt(
            directory_structure=dir_structure,
            dependencies=dependencies,
            languages=lang_str,
            config_files=config_files,
            readme=readme,
        )
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=PROJECT_ANALYSIS_SYSTEM),
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
            "type": "project_analysis",
            "content": response.content,
            "model": response.model,
        })
        
        return {
            "content": response.content,
            "model": response.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
    
    def _analyze_structure(self, project_path: Path) -> dict:
        """Analyze directory structure."""
        structure = {
            "directories": [],
            "file_count": 0,
            "total_lines": 0,
        }
        
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        
        for root, dirs, files in os.walk(project_path):
            # Exclude certain directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            rel_path = os.path.relpath(root, project_path)
            if rel_path != ".":
                structure["directories"].append(rel_path)
            
            for f in files:
                structure["file_count"] += 1
                file_path = Path(root) / f
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                        structure["total_lines"] += sum(1 for _ in fp)
                except (IOError, OSError):
                    pass
                    
        return structure
    
    def _analyze_dependencies(self, project_path: Path) -> dict:
        """Analyze project dependencies."""
        deps = {
            "python": None,
            "javascript": None,
            "go": None,
        }
        
        # Python dependencies
        requirements_file = project_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    deps["python"] = [
                        line.strip() for line in f 
                        if line.strip() and not line.startswith("#")
                    ]
            except (IOError, OSError):
                pass
                
        pyproject_file = project_path / "pyproject.toml"
        if pyproject_file.exists() and deps["python"] is None:
            deps["python"] = self._parse_pyproject(pyproject_file)
            
        # JavaScript dependencies
        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    pkg = json.load(f)
                    deps["javascript"] = {
                        "dependencies": pkg.get("dependencies", {}),
                        "devDependencies": pkg.get("devDependencies", {}),
                    }
            except (json.JSONDecodeError, IOError):
                pass
                
        # Go dependencies
        go_mod = project_path / "go.mod"
        if go_mod.exists():
            try:
                with open(go_mod, 'r') as f:
                    deps["go"] = f.read()
            except IOError:
                pass
                
        return deps
    
    def _parse_pyproject(self, pyproject_file: Path) -> Optional[list]:
        """Parse pyproject.toml for dependencies."""
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
                # Simple parsing - look for dependencies section
                if "[project]" in content and "dependencies" in content:
                    return ["pyproject.toml detected"]
        except IOError:
            pass
        return None
    
    def _detect_languages(self, project_path: Path) -> dict:
        """Detect programming languages used."""
        extensions = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
            ".rb": "Ruby",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
            ".html": "HTML",
            ".css": "CSS",
            ".md": "Markdown",
        }
        
        language_counts = {}
        exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in extensions:
                    lang = extensions[ext]
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    
        return language_counts
    
    def _detect_patterns(self, project_path: Path) -> dict:
        """Detect common project patterns and frameworks."""
        patterns = {
            "framework": None,
            "testing": [],
            "ci_cd": [],
            "containerization": False,
        }
        
        # Detect web frameworks
        if (project_path / "requirements.txt").exists():
            try:
                with open(project_path / "requirements.txt", 'r') as f:
                    content = f.read().lower()
                    if "fastapi" in content:
                        patterns["framework"] = "FastAPI"
                    elif "django" in content:
                        patterns["framework"] = "Django"
                    elif "flask" in content:
                        patterns["framework"] = "Flask"
            except IOError:
                pass
                
        if (project_path / "package.json").exists():
            try:
                with open(project_path / "package.json", 'r') as f:
                    pkg = json.load(f)
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    if "react" in deps:
                        patterns["framework"] = "React"
                    elif "vue" in deps:
                        patterns["framework"] = "Vue"
                    elif "express" in deps:
                        patterns["framework"] = "Express"
            except (json.JSONDecodeError, IOError):
                pass
        
        # Detect testing frameworks
        if (project_path / "pytest.ini").exists() or (project_path / "conftest.py").exists():
            patterns["testing"].append("pytest")
        if (project_path / "jest.config.js").exists():
            patterns["testing"].append("jest")
            
        # Detect CI/CD
        github_workflows = project_path / ".github" / "workflows"
        if github_workflows.exists():
            patterns["ci_cd"].append("GitHub Actions")
            
        # Detect containerization
        if (project_path / "Dockerfile").exists():
            patterns["containerization"] = True
            
        return patterns
    
    def _read_readme(self, project_path: Path) -> Optional[str]:
        """Read README content."""
        readme_names = ["README.md", "README.rst", "README.txt", "README"]
        
        for readme_name in readme_names:
            readme_path = project_path / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Truncate if too long
                        if len(content) > 5000:
                            return content[:5000] + "\n... (truncated)"
                        return content
                except IOError:
                    pass
        return None
    
    def _find_config_files(self, project_path: Path) -> list:
        """Find configuration files."""
        config_patterns = [
            "*.toml", "*.yaml", "*.yml", "*.json", "*.ini", "*.cfg",
            ".eslintrc*", ".prettierrc*", "tsconfig.json", "setup.py",
            "setup.cfg", "Makefile", "docker-compose*",
        ]
        
        config_files = []
        for pattern in config_patterns:
            if "*" in pattern:
                # Simple glob matching
                prefix = pattern.split("*")[0]
                for f in project_path.iterdir():
                    if f.is_file() and f.name.startswith(prefix):
                        config_files.append(f.name)
            else:
                if (project_path / pattern).exists():
                    config_files.append(pattern)
                    
        return list(set(config_files))
