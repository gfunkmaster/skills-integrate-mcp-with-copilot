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
- **Project Analysis**: Automatically understand project structure, dependencies, languages, and patterns
- **Issue Analysis**: Parse and classify issues to extract actionable requirements
- **Code Review**: Identify relevant files and potential code quality issues
- **Solution Implementation**: Generate detailed implementation plans with risk assessment
- **LLM Integration**: Use AI-powered code generation with OpenAI or Anthropic

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
For LLM support, install the optional dependencies:

```bash
pip install openai      # For OpenAI support
pip install anthropic   # For Anthropic Claude support
```

## Quick Start

### Basic Usage (Static Analysis)

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
### With LLM Integration (AI-Powered Solutions)

```python
from agentic_chain import AgenticChain, LLMFactory

# Create an LLM provider (uses OPENAI_API_KEY env var)
llm = LLMFactory.create("openai", model="gpt-4")

# Or use Anthropic Claude (uses ANTHROPIC_API_KEY env var)
# llm = LLMFactory.create("anthropic", model="claude-3-sonnet-20240229")

# Initialize the chain with LLM support
chain = AgenticChain(
    project_path="/path/to/your/project",
    llm_provider=llm
)

# Solve an issue with AI-generated code suggestions
result = chain.solve_issue(issue)

# View LLM usage statistics
print(chain.get_llm_usage())
# {'provider': 'openai', 'model': 'gpt-4', 'total_tokens': 1500, 'estimated_cost': 0.045}
```

### Alternative: Configure LLM via Dictionary

```python
chain = AgenticChain(
    project_path="/path/to/project",
    llm_config={
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "your-api-key",  # Optional: uses env var if not set
    }
)
```

### Command Line Interface

```bash
# Quick issue analysis
agentic-chain solve /path/to/project --title "Fix login bug" --body "Users cannot login..."

# Analyze with full output
agentic-chain solve /path/to/project --issue-file issue.json --summary

# Export for automation
# Solve an issue with inline data
agentic-chain solve /path/to/project --title "Fix login bug" --body "Users cannot login..."

# Use LLM for AI-powered solutions
agentic-chain solve /path/to/project --issue-file issue.json --llm openai --summary

# Use a specific model
agentic-chain solve /path/to/project --issue-file issue.json --llm anthropic --model claude-3-sonnet-20240229

# Show LLM token usage
agentic-chain solve /path/to/project --issue-file issue.json --llm openai --show-usage

# Export results
agentic-chain solve /path/to/project --issue-file issue.json --output result.json

# List available LLM providers
agentic-chain providers
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
- **AI-generated code suggestions** (when LLM is enabled)
- **AI-powered implementation plans** (when LLM is enabled)

## LLM Integration

The framework supports multiple LLM providers for intelligent code generation:

### Supported Providers

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| OpenAI | gpt-4, gpt-4-turbo, gpt-3.5-turbo | `OPENAI_API_KEY` |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku | `ANTHROPIC_API_KEY` |

### Features

- **Streaming responses**: Get output as it's generated
- **Token tracking**: Monitor usage and estimated costs
- **Automatic retries**: Handle rate limits gracefully
- **Fallback mechanism**: Works without LLM (static analysis)

### Usage Tracking

```python
from agentic_chain import AgenticChain, LLMFactory

llm = LLMFactory.create("openai")
chain = AgenticChain(project_path="/project", llm_provider=llm)
chain.solve_issue(issue)

# Get usage stats
usage = chain.get_llm_usage()
print(f"Tokens used: {usage['total_tokens']}")
print(f"Estimated cost: ${usage['estimated_cost']:.4f}")
```

### Custom LLM Provider

```python
from agentic_chain.llm import LLMProvider, LLMConfig, LLMResponse, LLMMessage

class CustomProvider(LLMProvider):
    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # Your implementation
        pass
    
    def stream(self, messages: list[LLMMessage], **kwargs):
        # Your implementation
        pass

# Register the provider
from agentic_chain.llm import LLMFactory
LLMFactory.register_provider("custom", CustomProvider)
```

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
        "risks": [ ... ],
        "llm_generated": true,
        "ai_implementation_plan": { ... },
        "code_suggestions": { ... }
    },
    "llm_usage": {
        "provider": "openai",
        "model": "gpt-4",
        "total_tokens": 1500,
        "estimated_cost": 0.045
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
- **Automated Issue Triage**: Classify and prioritize incoming issues
- **AI-Powered Code Generation**: Generate actual code solutions
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
