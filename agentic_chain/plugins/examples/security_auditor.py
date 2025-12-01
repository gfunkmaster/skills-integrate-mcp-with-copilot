"""
Security Auditor Plugin - Scans for security vulnerabilities.

This plugin demonstrates how to create a custom agent that analyzes
code for potential security issues.
"""

import logging
import os
import re
from pathlib import Path
from typing import List

from ...agents import BaseAgent, AgentContext


logger = logging.getLogger(__name__)


class SecurityAuditor(BaseAgent):
    """
    Scans project code for potential security vulnerabilities.
    
    This plugin checks for:
    - Hardcoded secrets and credentials
    - SQL injection vulnerabilities
    - Command injection risks
    - Unsafe deserialization
    - Insecure dependencies (if requirements.txt exists)
    
    Example:
        from agentic_chain.plugins.examples import SecurityAuditor
        
        auditor = SecurityAuditor()
        context = auditor.execute(context)
        
        # Results are stored in context.plugin_results["security_audit"]
        findings = context.plugin_results.get("security_audit", {})
    """
    
    # Patterns that may indicate security issues
    SECRET_PATTERNS = [
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        (r'token\s*=\s*["\'][A-Za-z0-9_\-]+["\']', "Hardcoded token"),
        (r'AWS_ACCESS_KEY_ID\s*=\s*["\'][^"\']+["\']', "Hardcoded AWS key"),
        (r'AWS_SECRET_ACCESS_KEY\s*=\s*["\'][^"\']+["\']', "Hardcoded AWS secret"),
    ]
    
    SQL_INJECTION_PATTERNS = [
        (r'execute\([^)]*%[sd]', "Potential SQL injection (string formatting)"),
        (r'execute\([^)]*\+\s*\w+', "Potential SQL injection (string concatenation)"),
        (r'f["\'].*SELECT.*{', "Potential SQL injection (f-string)"),
        (r'f["\'].*INSERT.*{', "Potential SQL injection (f-string)"),
        (r'f["\'].*UPDATE.*{', "Potential SQL injection (f-string)"),
        (r'f["\'].*DELETE.*{', "Potential SQL injection (f-string)"),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        (r'os\.system\([^)]*\+', "Potential command injection (os.system)"),
        (r'subprocess\.(call|run|Popen)\([^)]*shell\s*=\s*True', "Shell=True with subprocess"),
        (r'eval\(', "Use of eval() - potential code injection"),
        (r'exec\(', "Use of exec() - potential code injection"),
    ]
    
    UNSAFE_PATTERNS = [
        (r'pickle\.loads?\(', "Unsafe pickle deserialization"),
        (r'yaml\.load\([^)]*\)', "Unsafe YAML loading (use safe_load)"),
        (r'verify\s*=\s*False', "SSL verification disabled"),
        (r'SECURITY_ENABLED\s*=\s*False', "Security explicitly disabled"),
    ]
    
    def __init__(self, name: str = "SecurityAuditor"):
        super().__init__(name)
        self._severity_weights = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 1,
        }
    
    @property
    def dependencies(self) -> List[str]:
        """SecurityAuditor can run after ProjectAnalyzer for better context."""
        return ["ProjectAnalyzer"]
    
    @property
    def description(self) -> str:
        return "Scans code for security vulnerabilities including hardcoded secrets, injection risks, and unsafe patterns"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute security audit on the project.
        
        Args:
            context: The agent context with project information.
            
        Returns:
            Updated context with security findings in plugin_results.
        """
        project_path = Path(context.project_path)
        
        findings = []
        
        # Scan Python files
        findings.extend(self._scan_directory(project_path, ['.py']))
        
        # Scan other common files
        findings.extend(self._scan_directory(project_path, ['.js', '.ts', '.jsx', '.tsx']))
        
        # Check for insecure dependencies
        dep_findings = self._check_dependencies(project_path)
        findings.extend(dep_findings)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(findings)
        
        # Categorize findings by severity
        findings_by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }
        for finding in findings:
            severity = finding.get("severity", "medium")
            findings_by_severity[severity].append(finding)
        
        audit_result = {
            "total_findings": len(findings),
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "findings": findings,
            "findings_by_severity": findings_by_severity,
            "summary": self._generate_summary(findings),
            "recommendations": self._generate_recommendations(findings),
        }
        
        context.plugin_results["security_audit"] = audit_result
        logger.info(f"Security audit complete: {len(findings)} findings, risk score: {risk_score}")
        
        return context
    
    def _scan_directory(self, directory: Path, extensions: List[str]) -> List[dict]:
        """Scan directory for security issues."""
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
                        lines = content.split('\n')
                        
                        # Check each pattern category
                        findings.extend(self._check_patterns(
                            str(rel_path), lines, self.SECRET_PATTERNS, "critical", "secret"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), lines, self.SQL_INJECTION_PATTERNS, "high", "sql_injection"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), lines, self.COMMAND_INJECTION_PATTERNS, "high", "command_injection"
                        ))
                        findings.extend(self._check_patterns(
                            str(rel_path), lines, self.UNSAFE_PATTERNS, "medium", "unsafe_code"
                        ))
                except (IOError, OSError) as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        return findings
    
    def _check_patterns(
        self,
        file_path: str,
        lines: List[str],
        patterns: List[tuple],
        severity: str,
        category: str,
    ) -> List[dict]:
        """Check lines against patterns."""
        findings = []
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                continue
            
            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "file": file_path,
                        "line": line_num,
                        "severity": severity,
                        "category": category,
                        "description": description,
                        "code_snippet": line.strip()[:100],
                    })
        return findings
    
    def _check_dependencies(self, project_path: Path) -> List[dict]:
        """Check for known vulnerable dependencies."""
        findings = []
        
        # Check requirements.txt
        requirements_file = project_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Check for pinned versions (security best practice)
                        if '==' not in line and '>=' not in line and '<=' not in line:
                            if '@' not in line and 'git+' not in line:
                                findings.append({
                                    "file": "requirements.txt",
                                    "line": line_num,
                                    "severity": "low",
                                    "category": "dependency",
                                    "description": f"Unpinned dependency: {line}",
                                    "code_snippet": line,
                                })
            except (IOError, OSError) as e:
                logger.warning(f"Could not read requirements.txt: {e}")
        
        return findings
    
    def _calculate_risk_score(self, findings: List[dict]) -> int:
        """Calculate overall risk score (0-100)."""
        if not findings:
            return 0
        
        score = 0
        for finding in findings:
            severity = finding.get("severity", "medium")
            score += self._severity_weights.get(severity, 4)
        
        # Normalize to 0-100
        return min(100, score)
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level from score."""
        if score >= 50:
            return "critical"
        elif score >= 30:
            return "high"
        elif score >= 10:
            return "medium"
        else:
            return "low"
    
    def _generate_summary(self, findings: List[dict]) -> str:
        """Generate a human-readable summary."""
        if not findings:
            return "No security issues found."
        
        categories = {}
        for finding in findings:
            cat = finding.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1
        
        parts = []
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            parts.append(f"{count} {cat.replace('_', ' ')} issue(s)")
        
        return f"Found {len(findings)} security issues: " + ", ".join(parts)
    
    def _generate_recommendations(self, findings: List[dict]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        categories = {f.get("category") for f in findings}
        
        if "secret" in categories:
            recommendations.append("Move hardcoded secrets to environment variables or a secrets manager")
        if "sql_injection" in categories:
            recommendations.append("Use parameterized queries or an ORM to prevent SQL injection")
        if "command_injection" in categories:
            recommendations.append("Avoid shell=True with subprocess; validate and sanitize all inputs")
        if "unsafe_code" in categories:
            recommendations.append("Review and fix unsafe code patterns (eval, exec, pickle, etc.)")
        if "dependency" in categories:
            recommendations.append("Pin dependency versions and regularly update them")
        
        if not recommendations:
            recommendations.append("Continue following security best practices")
        
        return recommendations
