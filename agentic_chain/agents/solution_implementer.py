"""
Solution Implementer Agent - Generates and applies code fixes.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from . import BaseAgent, AgentContext


logger = logging.getLogger(__name__)


class SolutionImplementer(BaseAgent):
    """
    Generates solution proposals and implementation plans based on
    issue analysis and code review.
    
    When an LLM provider is configured, uses AI to generate actual code
    solutions and intelligent implementation plans.
    """
    
    def __init__(
        self,
        name: str = "SolutionImplementer",
        llm_provider: Optional["LLMProvider"] = None,
    ):
        """
        Initialize the SolutionImplementer.
        
        Args:
            name: Agent name.
            llm_provider: Optional LLM provider for AI-powered generation.
        """
        super().__init__(name)
        self._llm_provider = llm_provider
        
    @property
    def llm_provider(self):
        """Get the LLM provider."""
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider):
        """Set the LLM provider."""
        self._llm_provider = provider
        
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Generate solution proposals and implementation plan.
        
        If an LLM provider is available, generates AI-powered solutions.
        Otherwise, falls back to static analysis.
        """
        if not context.issue_analysis:
            raise ValueError("Issue analysis is required to implement solution")
        
        # Check if LLM is available
        if self._llm_provider:
            logger.info("Using LLM-powered solution generation")
            solution = self._generate_llm_solution(context)
        else:
            logger.info("Using static analysis for solution generation")
            solution = self._generate_static_solution(context)
        
        context.solution = solution
        return context
    
    def _generate_llm_solution(self, context: AgentContext) -> dict:
        """Generate solution using LLM."""
        solution = {
            "proposed_changes": self._generate_proposed_changes(context),
            "implementation_plan": self._create_implementation_plan(context),
            "test_strategy": self._define_test_strategy(context),
            "documentation_updates": self._identify_doc_updates(context),
            "risks": self._assess_risks(context),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_generated": True,
        }
        
        # Generate AI-powered implementation plan
        try:
            ai_plan = self._generate_ai_implementation_plan(context)
            if ai_plan:
                solution["ai_implementation_plan"] = ai_plan
        except Exception as e:
            logger.warning(f"Failed to generate AI implementation plan: {e}")
        
        # Generate AI-powered code suggestions
        try:
            code_suggestions = self._generate_ai_code_suggestions(context)
            if code_suggestions:
                solution["code_suggestions"] = code_suggestions
        except Exception as e:
            logger.warning(f"Failed to generate AI code suggestions: {e}")
        
        return solution
    
    def _generate_static_solution(self, context: AgentContext) -> dict:
        """Generate solution using static analysis (fallback)."""
        return {
            "proposed_changes": self._generate_proposed_changes(context),
            "implementation_plan": self._create_implementation_plan(context),
            "test_strategy": self._define_test_strategy(context),
            "documentation_updates": self._identify_doc_updates(context),
            "risks": self._assess_risks(context),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_generated": False,
        }
    
    def _generate_ai_implementation_plan(self, context: AgentContext) -> Optional[dict]:
        """Generate an AI-powered implementation plan."""
        if not self._llm_provider:
            return None
        
        issue_data = context.issue_data or {}
        issue_description = f"{issue_data.get('title', '')}\n\n{issue_data.get('body', '')}"
        
        response = self._llm_provider.generate_implementation_plan(
            issue_description=issue_description,
            project_context=context.project_analysis,
        )
        
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
            "type": "implementation_plan",
            "content": response.content,
            "model": response.model,
        })
        
        return {
            "content": response.content,
            "model": response.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
    
    def _generate_ai_code_suggestions(self, context: AgentContext) -> Optional[dict]:
        """Generate AI-powered code suggestions."""
        if not self._llm_provider:
            return None
        
        issue_analysis = context.issue_analysis or {}
        code_review = context.code_review or {}
        
        # Build prompt with context
        issue_type = issue_analysis.get("issue_type", "unknown")
        requirements = issue_analysis.get("requirements", [])
        relevant_files = code_review.get("relevant_files", [])
        
        prompt = f"""Based on the following issue analysis, generate code suggestions:

Issue Type: {issue_type}
Requirements:
{chr(10).join('- ' + req for req in requirements[:5])}

Relevant files: {', '.join(relevant_files[:5]) if relevant_files else 'None identified'}

Please provide specific code changes or new code that would address these requirements."""

        # Detect primary language from project
        language = None
        if context.project_analysis:
            languages = context.project_analysis.get("languages", {})
            if languages:
                language = max(languages.items(), key=lambda x: x[1])[0]
        
        response = self._llm_provider.generate_code(
            prompt=prompt,
            language=language,
        )
        
        # Track usage
        if response.usage:
            context.llm_context.add_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost=response.usage.estimated_cost,
            )
        
        context.llm_context.responses.append({
            "type": "code_suggestion",
            "content": response.content,
            "model": response.model,
        })
        
        return {
            "content": response.content,
            "model": response.model,
            "language": language,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
    
    def _generate_proposed_changes(self, context: AgentContext) -> list:
        """Generate list of proposed code changes."""
        changes = []
        
        issue_analysis = context.issue_analysis
        code_review = context.code_review or {}
        
        issue_type = issue_analysis.get("issue_type", "unknown")
        requirements = issue_analysis.get("requirements", [])
        relevant_files = code_review.get("relevant_files", [])
        
        # Generate changes based on issue type
        if issue_type == "bug":
            changes.append({
                "type": "fix",
                "description": "Identify and fix the root cause of the bug",
                "affected_files": relevant_files[:5],
                "steps": [
                    "Reproduce the bug locally",
                    "Identify the faulty code",
                    "Implement the fix",
                    "Verify the fix works",
                    "Add regression test",
                ],
            })
            
        elif issue_type == "feature":
            changes.append({
                "type": "implementation",
                "description": "Implement the new feature",
                "affected_files": relevant_files[:5],
                "steps": [
                    "Design the feature architecture",
                    "Implement core functionality",
                    "Add necessary API endpoints",
                    "Implement UI if needed",
                    "Add comprehensive tests",
                    "Update documentation",
                ],
            })
            
        elif issue_type == "enhancement":
            changes.append({
                "type": "enhancement",
                "description": "Improve existing functionality",
                "affected_files": relevant_files[:5],
                "steps": [
                    "Review current implementation",
                    "Identify improvement areas",
                    "Implement enhancements",
                    "Ensure backward compatibility",
                    "Update tests",
                ],
            })
            
        elif issue_type == "documentation":
            changes.append({
                "type": "documentation",
                "description": "Update documentation",
                "affected_files": ["README.md", "docs/"],
                "steps": [
                    "Identify documentation gaps",
                    "Write or update content",
                    "Add examples if needed",
                    "Review for clarity",
                ],
            })
            
        else:
            changes.append({
                "type": "investigation",
                "description": "Investigate and implement solution",
                "affected_files": relevant_files[:5],
                "steps": [
                    "Understand the requirement",
                    "Investigate current implementation",
                    "Design solution",
                    "Implement changes",
                    "Test thoroughly",
                ],
            })
            
        # Add requirement-specific changes
        for i, req in enumerate(requirements[:5]):
            changes.append({
                "type": "requirement",
                "description": f"Implement: {req[:100]}",
                "priority": i + 1,
                "status": "pending",
            })
            
        return changes
    
    def _create_implementation_plan(self, context: AgentContext) -> dict:
        """Create a detailed implementation plan."""
        issue_analysis = context.issue_analysis
        priority = issue_analysis.get("priority", "medium")
        
        # Estimate complexity based on requirements count
        requirements = issue_analysis.get("requirements", [])
        if len(requirements) <= 2:
            complexity = "low"
            estimated_hours = 2
        elif len(requirements) <= 5:
            complexity = "medium"
            estimated_hours = 8
        else:
            complexity = "high"
            estimated_hours = 24
            
        plan = {
            "priority": priority,
            "complexity": complexity,
            "estimated_hours": estimated_hours,
            "phases": [
                {
                    "name": "Analysis",
                    "description": "Understand requirements and codebase",
                    "estimated_hours": estimated_hours * 0.2,
                    "tasks": [
                        "Review issue requirements",
                        "Analyze relevant code",
                        "Identify dependencies",
                    ],
                },
                {
                    "name": "Design",
                    "description": "Design the solution",
                    "estimated_hours": estimated_hours * 0.2,
                    "tasks": [
                        "Design solution architecture",
                        "Identify edge cases",
                        "Plan test coverage",
                    ],
                },
                {
                    "name": "Implementation",
                    "description": "Implement the solution",
                    "estimated_hours": estimated_hours * 0.4,
                    "tasks": [
                        "Write code changes",
                        "Implement tests",
                        "Handle edge cases",
                    ],
                },
                {
                    "name": "Testing",
                    "description": "Test and validate",
                    "estimated_hours": estimated_hours * 0.15,
                    "tasks": [
                        "Run unit tests",
                        "Perform integration testing",
                        "Manual verification",
                    ],
                },
                {
                    "name": "Review",
                    "description": "Code review and refinement",
                    "estimated_hours": estimated_hours * 0.05,
                    "tasks": [
                        "Self-review code",
                        "Address review comments",
                        "Final cleanup",
                    ],
                },
            ],
            "acceptance_criteria": issue_analysis.get("acceptance_criteria", []),
        }
        
        return plan
    
    def _define_test_strategy(self, context: AgentContext) -> dict:
        """Define testing strategy for the solution."""
        code_review = context.code_review or {}
        quality = code_review.get("code_quality", {})
        project_analysis = context.project_analysis or {}
        
        strategy = {
            "unit_tests": {
                "required": True,
                "coverage_target": 80,
                "framework": self._detect_test_framework(project_analysis),
            },
            "integration_tests": {
                "required": True,
                "scenarios": [],
            },
            "manual_testing": {
                "required": True,
                "test_cases": [],
            },
        }
        
        # Add test scenarios based on issue
        if context.issue_analysis:
            requirements = context.issue_analysis.get("requirements", [])
            for req in requirements[:5]:
                strategy["integration_tests"]["scenarios"].append(
                    f"Verify: {req[:50]}"
                )
                strategy["manual_testing"]["test_cases"].append(
                    f"Test: {req[:50]}"
                )
                
        return strategy
    
    def _detect_test_framework(self, project_analysis: dict) -> str:
        """Detect the testing framework used."""
        if project_analysis.get("patterns", {}).get("testing"):
            frameworks = project_analysis["patterns"]["testing"]
            return frameworks[0] if frameworks else "pytest"
            
        languages = project_analysis.get("languages", {})
        if "Python" in languages:
            return "pytest"
        elif "JavaScript" in languages or "TypeScript" in languages:
            return "jest"
        elif "Go" in languages:
            return "go test"
        elif "Java" in languages:
            return "JUnit"
            
        return "pytest"  # Default
    
    def _identify_doc_updates(self, context: AgentContext) -> list:
        """Identify documentation that needs updating."""
        updates = []
        
        issue_analysis = context.issue_analysis
        issue_type = issue_analysis.get("issue_type", "")
        
        if issue_type == "feature":
            updates.extend([
                {"file": "README.md", "section": "Features", "action": "Add new feature description"},
                {"file": "docs/", "section": "API", "action": "Document new endpoints"},
            ])
            
        elif issue_type == "bug":
            updates.append({
                "file": "CHANGELOG.md",
                "section": "Bug Fixes",
                "action": "Document the bug fix",
            })
            
        elif issue_type == "documentation":
            updates.append({
                "file": "README.md",
                "section": "Documentation",
                "action": "Update as per issue requirements",
            })
            
        # Always update changelog
        updates.append({
            "file": "CHANGELOG.md",
            "section": "Changes",
            "action": "Document this change",
        })
        
        return updates
    
    def _assess_risks(self, context: AgentContext) -> list:
        """Assess risks associated with the implementation."""
        risks = []
        
        issue_analysis = context.issue_analysis
        code_review = context.code_review or {}
        
        # Check for breaking changes
        if code_review.get("relevant_files"):
            if len(code_review["relevant_files"]) > 10:
                risks.append({
                    "level": "medium",
                    "description": "Large number of files affected - higher chance of side effects",
                    "mitigation": "Thorough testing and incremental changes",
                })
                
        # Check complexity
        if issue_analysis.get("requirements") and len(issue_analysis["requirements"]) > 5:
            risks.append({
                "level": "medium",
                "description": "Multiple requirements increase implementation complexity",
                "mitigation": "Break down into smaller tasks",
            })
            
        # Check for security concerns
        keywords = issue_analysis.get("keywords", [])
        security_keywords = {"auth", "password", "token", "security", "permission", "access"}
        if any(kw in security_keywords for kw in keywords):
            risks.append({
                "level": "high",
                "description": "Security-sensitive changes require careful review",
                "mitigation": "Security review and penetration testing",
            })
            
        # Check for database/data changes
        data_keywords = {"database", "migration", "schema", "data", "storage"}
        if any(kw in data_keywords for kw in keywords):
            risks.append({
                "level": "medium",
                "description": "Data-related changes may affect existing data",
                "mitigation": "Backup data and test migrations",
            })
            
        # Default low risk if no issues found
        if not risks:
            risks.append({
                "level": "low",
                "description": "Standard implementation with minimal risk",
                "mitigation": "Standard testing procedures",
            })
            
        return risks
