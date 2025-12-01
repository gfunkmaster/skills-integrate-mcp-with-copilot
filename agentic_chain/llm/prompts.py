"""
LLM Prompt Templates - Standardized prompts for agentic chain operations.

This module provides prompt templates for different agent operations:
- Project analysis and architecture understanding
- Issue classification and requirement extraction
- Code review and suggestions
- Implementation planning
"""

from typing import Optional


# =============================================================================
# Project Analysis Prompts
# =============================================================================

PROJECT_ANALYSIS_SYSTEM = """You are an expert software architect analyzing a codebase.
Your goal is to understand the project's architecture, patterns, and key components.

Be concise and focus on actionable insights. Structure your response clearly."""

PROJECT_ANALYSIS_PROMPT = """Analyze this software project based on the following information:

## Directory Structure
{directory_structure}

## Dependencies
{dependencies}

## Detected Languages
{languages}

## Configuration Files
{config_files}

{readme_section}

Provide a brief analysis covering:
1. **Project Type**: What kind of project is this? (web app, library, CLI tool, etc.)
2. **Architecture Pattern**: Identify any architectural patterns (MVC, microservices, etc.)
3. **Key Components**: List the main components and their purposes
4. **Technology Stack**: Summarize the tech stack
5. **Code Organization**: Comment on the project structure quality

Keep your response focused and actionable."""


# =============================================================================
# Issue Analysis Prompts
# =============================================================================

ISSUE_CLASSIFICATION_SYSTEM = """You are an expert at analyzing GitHub issues.
Your goal is to classify issues, extract requirements, and assess priority.

Be precise and structured in your analysis. Use the exact format requested."""

ISSUE_CLASSIFICATION_PROMPT = """Analyze this GitHub issue and provide a structured classification:

## Issue
**Title**: {issue_title}
**Body**: 
{issue_body}

**Labels**: {labels}

{project_context_section}

Provide your analysis in the following JSON format:
```json
{{
    "issue_type": "bug|feature|enhancement|documentation|security|performance|unknown",
    "priority": "critical|high|medium|low",
    "complexity": "low|medium|high",
    "estimated_hours": <number>,
    "summary": "<one-sentence summary of the issue>",
    "key_requirements": ["<requirement 1>", "<requirement 2>", ...],
    "affected_areas": ["<component/file/area 1>", ...],
    "suggested_approach": "<brief implementation approach>",
    "potential_risks": ["<risk 1>", ...],
    "clarification_needed": ["<question 1>", ...] or []
}}
```

Be accurate and thorough. If information is missing, make reasonable inferences but note uncertainty."""


# =============================================================================
# Code Review Prompts
# =============================================================================

CODE_REVIEW_SYSTEM = """You are an expert code reviewer with deep knowledge of software best practices.
Focus on:
- Potential bugs and logic errors
- Security vulnerabilities
- Performance issues
- Code quality and maintainability
- Adherence to best practices

Be constructive and specific in your feedback."""

CODE_REVIEW_PROMPT = """Review the following code in the context of this issue:

## Issue Context
{issue_summary}

## Code to Review
```{language}
{code}
```

## File: {file_path}

Provide your review in the following JSON format:
```json
{{
    "summary": "<brief summary of the code's purpose>",
    "quality_score": <1-10>,
    "issues": [
        {{
            "severity": "critical|high|medium|low|info",
            "type": "bug|security|performance|style|maintainability",
            "line": <line_number or null>,
            "description": "<description of the issue>",
            "suggestion": "<how to fix it>"
        }}
    ],
    "strengths": ["<positive aspect 1>", ...],
    "improvements": ["<suggested improvement 1>", ...]
}}
```

Focus on the most important issues. Be specific about line numbers when possible."""

CODE_REVIEW_SUMMARY_PROMPT = """Based on the analysis of {file_count} files, provide a summary of the code review:

## Relevant Files
{file_list}

## Key Findings
{key_findings}

## Issue Context
{issue_summary}

Provide an overall assessment and prioritized recommendations for addressing the issue.
Structure your response as:
1. Overall Code Quality Assessment
2. Key Issues Found (prioritized)
3. Recommended Approach
4. Suggested File Changes"""


# =============================================================================
# Implementation Planning Prompts
# =============================================================================

IMPLEMENTATION_PLAN_SYSTEM = """You are an expert software developer creating implementation plans.
Your plans should be:
- Specific and actionable
- Well-structured with clear phases
- Risk-aware with mitigation strategies
- Realistic in time estimates

Provide practical guidance that a developer can follow."""

IMPLEMENTATION_PLAN_PROMPT = """Create a detailed implementation plan for this issue:

## Issue
{issue_description}

## Issue Analysis
- Type: {issue_type}
- Priority: {priority}
- Requirements: {requirements}

## Project Context
{project_context}

## Relevant Files
{relevant_files}

## Code Review Findings
{code_review_summary}

Create an implementation plan in the following JSON format:
```json
{{
    "summary": "<one-paragraph summary of the solution>",
    "approach": "<high-level approach description>",
    "phases": [
        {{
            "name": "<phase name>",
            "description": "<what this phase accomplishes>",
            "estimated_hours": <number>,
            "tasks": [
                {{
                    "description": "<task description>",
                    "file": "<file to modify or create>",
                    "type": "create|modify|delete"
                }}
            ]
        }}
    ],
    "test_strategy": {{
        "unit_tests": ["<test case 1>", ...],
        "integration_tests": ["<test case 1>", ...],
        "manual_tests": ["<test case 1>", ...]
    }},
    "risks": [
        {{
            "description": "<risk description>",
            "severity": "high|medium|low",
            "mitigation": "<how to mitigate>"
        }}
    ],
    "dependencies": ["<dependency or prerequisite 1>", ...],
    "total_estimated_hours": <number>
}}
```

Be thorough but realistic. Focus on actionable steps."""


# =============================================================================
# Helper Functions
# =============================================================================

def format_project_analysis_prompt(
    directory_structure: str,
    dependencies: str,
    languages: str,
    config_files: str,
    readme: Optional[str] = None,
) -> str:
    """Format the project analysis prompt with provided context."""
    readme_section = ""
    if readme:
        # Truncate readme if too long
        max_readme_length = 2000
        truncated_readme = readme[:max_readme_length]
        if len(readme) > max_readme_length:
            truncated_readme += "\n... (truncated)"
        readme_section = f"\n## README\n{truncated_readme}"
    
    return PROJECT_ANALYSIS_PROMPT.format(
        directory_structure=directory_structure,
        dependencies=dependencies,
        languages=languages,
        config_files=config_files,
        readme_section=readme_section,
    )


def format_issue_classification_prompt(
    issue_title: str,
    issue_body: str,
    labels: str = "None",
    project_context: Optional[str] = None,
) -> str:
    """Format the issue classification prompt with provided context."""
    project_context_section = ""
    if project_context:
        project_context_section = f"\n## Project Context\n{project_context}"
    
    return ISSUE_CLASSIFICATION_PROMPT.format(
        issue_title=issue_title,
        issue_body=issue_body or "No description provided",
        labels=labels,
        project_context_section=project_context_section,
    )


def format_code_review_prompt(
    code: str,
    file_path: str,
    issue_summary: str,
    language: str = "",
) -> str:
    """Format the code review prompt with provided context."""
    return CODE_REVIEW_PROMPT.format(
        code=code,
        file_path=file_path,
        issue_summary=issue_summary,
        language=language,
    )


def format_implementation_plan_prompt(
    issue_description: str,
    issue_type: str,
    priority: str,
    requirements: str,
    project_context: str,
    relevant_files: str,
    code_review_summary: str,
) -> str:
    """Format the implementation plan prompt with provided context."""
    return IMPLEMENTATION_PLAN_PROMPT.format(
        issue_description=issue_description,
        issue_type=issue_type,
        priority=priority,
        requirements=requirements,
        project_context=project_context,
        relevant_files=relevant_files,
        code_review_summary=code_review_summary,
    )
