# Agentic Chain

**The fastest way to get AI-powered insights on GitHub issues - under 5 seconds, zero configuration.**

Stop spending hours triaging issues. Get AI-powered insights in seconds.

## Quick Start

See the [full documentation](agentic_chain/README.md) for detailed installation and usage instructions.

### Installation

```bash
pip install agentic-chain
```

### Basic Usage

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

## Key Features

- **⚡ Instant Issue Classification**: Automatically categorize issues as bugs, features, enhancements, or documentation
- **📊 Priority Scoring**: Smart priority scoring algorithm based on labels, keywords, and urgency indicators
- **🔍 Similar Issue Detection**: Find potentially related or duplicate issues
- **⏱️ Time-to-Fix Estimation**: Estimate implementation effort based on complexity analysis
- **🏷️ Auto-Labeling Suggestions**: Get intelligent label recommendations for new issues
- **📈 Sentiment Analysis**: Detect urgency from issue language and tone
- **🔗 GitHub Native**: Deep integration with GitHub issue workflows

## Documentation

- [Full Documentation](agentic_chain/README.md)
- [Competitive Analysis](docs/competitive-analysis.md)
- [Plugin Guide](agentic_chain/plugins/PLUGIN_GUIDE.md)

## License

MIT License - see LICENSE file for details.

