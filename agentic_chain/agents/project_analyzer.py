"""
Project Analyzer Agent - Understands project structure, dependencies, and patterns.
"""

import os
import json
from pathlib import Path
from typing import Optional

from . import BaseAgent, AgentContext


class ProjectAnalyzer(BaseAgent):
    """
    Analyzes a project to understand its structure, dependencies,
    coding patterns, and conventions.
    """
    
    def __init__(self, name: str = "ProjectAnalyzer"):
        super().__init__(name)
        
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Analyze the project and populate context with analysis results.
        """
        if not self.validate_context(context):
            raise ValueError("Invalid context: project_path is required")
            
        project_path = Path(context.project_path)
        
        analysis = {
            "structure": self._analyze_structure(project_path),
            "dependencies": self._analyze_dependencies(project_path),
            "languages": self._detect_languages(project_path),
            "patterns": self._detect_patterns(project_path),
            "readme": self._read_readme(project_path),
            "config_files": self._find_config_files(project_path),
        }
        
        context.project_analysis = analysis
        return context
    
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
