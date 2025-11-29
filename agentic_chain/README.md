# Agentic Chain

An AI-powered agentic chain framework for solving issues in software projects. This package provides a modular, extensible pipeline of agents that work together to understand project context, analyze issues, review code, and propose solutions.

## Features

- **Project Analysis**: Automatically understand project structure, dependencies, languages, and patterns
- **Issue Analysis**: Parse and classify issues to extract actionable requirements
- **Code Review**: Identify relevant files and potential code quality issues
- **Solution Implementation**: Generate detailed implementation plans with risk assessment

## Installation

```bash
pip install agentic-chain
```

Or install from source:

```bash
git clone https://github.com/skills/integrate-mcp-with-copilot.git
cd integrate-mcp-with-copilot
pip install -e .
```

## Quick Start

### Python API

```python
from agentic_chain import AgenticChain

# Initialize the chain with your project path
chain = AgenticChain(project_path="/path/to/your/project")

# Define the issue to solve
issue = {
    "title": "Add user authentication feature",
    "body": "We need to implement user login and registration...",
    "labels": [{"name": "feature"}, {"name": "high-priority"}]
}

# Run the chain
result = chain.solve_issue(issue)

# Get a human-readable summary
print(chain.get_solution_summary())

# Export results to JSON
chain.export_result("solution.json")
```

### Command Line Interface

```bash
# Analyze a project
agentic-chain analyze /path/to/project

# Solve an issue from a JSON file
agentic-chain solve /path/to/project --issue-file issue.json --summary

# Solve an issue with inline data
agentic-chain solve /path/to/project --title "Fix login bug" --body "Users cannot login..."

# Export results
agentic-chain solve /path/to/project --issue-file issue.json --output result.json
```

## Architecture

The Agentic Chain consists of four main agents:

### 1. ProjectAnalyzer

Analyzes the project to understand:
- Directory structure and file counts
- Programming languages used
- Dependencies (Python, JavaScript, Go)
- Frameworks and patterns (FastAPI, React, etc.)
- Configuration files and CI/CD setup

### 2. IssueAnalyzer

Parses issues to extract:
- Issue type (bug, feature, enhancement, documentation)
- Priority level
- Affected files mentioned in the issue
- Requirements and acceptance criteria
- Keywords for context understanding

### 3. CodeReviewer

Reviews code to identify:
- Files relevant to the issue
- Code quality metrics (tests, docs, type hints)
- Potential issues (TODOs, FIXMEs, etc.)
- Improvement suggestions

### 4. SolutionImplementer

Generates solutions including:
- Proposed code changes with step-by-step guidance
- Implementation plan with time estimates
- Test strategy and coverage requirements
- Documentation updates needed
- Risk assessment and mitigation strategies

## Custom Agents

You can extend the chain with custom agents:

```python
from agentic_chain.agents import BaseAgent, AgentContext

class SecurityAuditor(BaseAgent):
    def __init__(self):
        super().__init__(name="SecurityAuditor")
    
    def execute(self, context: AgentContext) -> AgentContext:
        # Your custom logic here
        context.metadata["security_audit"] = {"status": "passed"}
        return context

# Add to chain
chain = AgenticChain(project_path="/path/to/project")
chain.add_agent(SecurityAuditor())
```

## Output Format

The chain produces a comprehensive result containing:

```json
{
    "project_path": "/path/to/project",
    "issue_data": { ... },
    "project_analysis": {
        "structure": { ... },
        "dependencies": { ... },
        "languages": { ... },
        "patterns": { ... }
    },
    "issue_analysis": {
        "title": "...",
        "issue_type": "bug",
        "priority": "high",
        "requirements": [ ... ]
    },
    "code_review": {
        "relevant_files": [ ... ],
        "code_quality": { ... },
        "potential_issues": [ ... ]
    },
    "solution": {
        "proposed_changes": [ ... ],
        "implementation_plan": { ... },
        "test_strategy": { ... },
        "risks": [ ... ]
    }
}
```

## Use Cases

- **Automated Issue Triage**: Classify and prioritize incoming issues
- **Code Review Automation**: Identify areas needing attention
- **Sprint Planning**: Estimate complexity and effort for issues
- **Onboarding**: Help new developers understand project structure
- **CI/CD Integration**: Generate implementation plans for automated workflows

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=agentic_chain
```

## License

MIT License - see LICENSE file for details.
