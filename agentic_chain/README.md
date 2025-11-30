# Agentic Chain

**The fastest way to get AI-powered insights on GitHub issues - under 5 seconds, zero configuration.**

Stop spending hours triaging issues. Get AI-powered insights in seconds.

## Why Agentic Chain?

| Feature | Agentic Chain | Heavy Frameworks |
|---------|---------------|------------------|
| Analysis Time | **< 5 seconds** | Minutes |
| Package Size | **< 5MB** | 50MB+ |
| Dependencies | **< 10** | 50+ |
| Setup Time | **< 1 minute** | Hours |
| Configuration | **Zero config** | Complex YAML |

## Key Features

- **⚡ Instant Issue Classification**: Automatically categorize issues as bugs, features, enhancements, or documentation
- **📊 Priority Scoring**: Smart priority scoring algorithm based on labels, keywords, and urgency indicators
- **🔍 Similar Issue Detection**: Find potentially related or duplicate issues
- **⏱️ Time-to-Fix Estimation**: Estimate implementation effort based on complexity analysis
- **🏷️ Auto-Labeling Suggestions**: Get intelligent label recommendations for new issues
- **📈 Sentiment Analysis**: Detect urgency from issue language and tone
- **🔗 GitHub Native**: Deep integration with GitHub issue workflows

## Target Audience

- ✅ Small to medium development teams
- ✅ Open source maintainers with high issue volume
- ✅ Solo developers needing quick issue triage
- ✅ Projects that find heavy frameworks too complex

## Technical Goals

- **Analysis time**: < 5 seconds per issue
- **Package size**: < 5MB
- **Memory usage**: < 100MB
- **Dependencies**: < 10 direct dependencies
- **Setup time**: < 1 minute

## Installation

```bash
pip install agentic-chain
```

That's it! No configuration files needed.

Or install from source:

```bash
git clone https://github.com/skills/integrate-mcp-with-copilot.git
cd integrate-mcp-with-copilot
pip install -e .
```

## Quick Start (< 1 minute)

### Analyze an Issue in Seconds

```python
from agentic_chain import AgenticChain

# One line setup
chain = AgenticChain(project_path="/path/to/your/project")

# Analyze any GitHub issue
issue = {
    "title": "Add user authentication feature",
    "body": "We need to implement user login and registration...",
    "labels": [{"name": "feature"}, {"name": "high-priority"}]
}

# Get insights in < 5 seconds
result = chain.solve_issue(issue)

# View the analysis
print(chain.get_solution_summary())
```

### Command Line (Zero Config)

```bash
# Quick issue analysis
agentic-chain solve /path/to/project --title "Fix login bug" --body "Users cannot login..."

# Analyze with full output
agentic-chain solve /path/to/project --issue-file issue.json --summary

# Export for automation
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

### Perfect For:
- **🔥 Issue Triage at Scale**: Automatically classify and prioritize hundreds of incoming issues
- **⏰ Sprint Planning**: Quick complexity and effort estimates for backlog grooming
- **🔄 Duplicate Detection**: Find similar issues before creating duplicates
- **🏃 Fast Feedback**: Get instant insights on new issues as they're created
- **🤖 CI/CD Integration**: Automate issue analysis in your GitHub workflows

### Example Workflow

1. New issue created → Agentic Chain analyzes in < 5 seconds
2. Get priority score, suggested labels, and complexity estimate
3. Auto-assign to appropriate team member based on analysis
4. Include time-to-fix estimate in sprint planning

## Comparison

| Criteria | Agentic Chain | CrewAI | LangChain Agents |
|----------|---------------|--------|------------------|
| Setup Time | < 1 minute | 30+ minutes | 15+ minutes |
| Config Required | None | YAML config | Python config |
| Package Size | < 5MB | 100MB+ | 50MB+ |
| Focus | GitHub Issues | General AI | General AI |
| Learning Curve | Minimal | Steep | Moderate |

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
