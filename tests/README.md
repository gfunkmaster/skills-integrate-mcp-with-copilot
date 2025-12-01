# Testing Guide

This guide explains how to run and write tests for the Agentic Chain project.

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

Or install development dependencies from pyproject.toml:

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=agentic_chain --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

### Run Specific Test Files

```bash
# Run a specific test file
pytest tests/test_llm_openai.py

# Run tests in a specific directory
pytest tests/

# Run a specific test class
pytest tests/test_orchestrator.py::TestAgenticChain

# Run a specific test function
pytest tests/test_orchestrator.py::TestAgenticChain::test_init
```

### Run Tests with Verbose Output

```bash
pytest -v
```

### Run Tests and Stop on First Failure

```bash
pytest -x
```

### Run Only Failed Tests from Last Run

```bash
pytest --lf
```

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures and configuration
├── test_llm_base.py               # Tests for LLM base classes
├── test_llm_openai.py             # Tests for OpenAI provider
├── test_llm_anthropic.py          # Tests for Anthropic provider (if exists)
├── test_orchestrator.py           # Tests for main orchestrator
├── test_project_analyzer.py       # Tests for project analysis
├── test_issue_analyzer.py         # Tests for issue analysis
├── test_code_reviewer.py          # Tests for code review
├── test_solution_implementer.py   # Tests for solution implementation
└── test_integration.py            # Integration tests (if exists)
```

## Writing Tests

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

- `sample_project`: Creates a temporary project structure for testing
- `sample_issue`: Provides a sample GitHub issue
- `mock_llm_config`: Mocked LLM configuration
- `mock_llm_response`: Mocked LLM response
- `mock_openai_client`: Mocked OpenAI client
- `sample_code`: Sample code for testing
- `sample_llm_messages`: Sample LLM messages

### Example Test

```python
import pytest
from agentic_chain.agents import AgentContext
from agentic_chain.agents.project_analyzer import ProjectAnalyzer


class TestMyFeature:
    """Test cases for my feature."""
    
    def test_basic_functionality(self, sample_project):
        """Test basic functionality."""
        # Arrange
        analyzer = ProjectAnalyzer()
        context = AgentContext(project_path=str(sample_project))
        
        # Act
        result = analyzer.execute(context)
        
        # Assert
        assert result.project_analysis is not None
        assert "languages" in result.project_analysis
```

### Mocking LLM Providers

To test agents that use LLM providers without making real API calls:

```python
from unittest.mock import MagicMock
from agentic_chain.llm.base import LLMResponse, LLMUsage


def test_with_mocked_llm(sample_project):
    """Test agent with mocked LLM provider."""
    # Create mock provider
    mock_provider = MagicMock()
    mock_provider.generate_code.return_value = LLMResponse(
        content="def hello(): return 'world'",
        model="gpt-4",
        usage=LLMUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            estimated_cost=0.001,
        ),
    )
    
    # Use in your test
    agent = SolutionImplementer(llm_provider=mock_provider)
    # ... rest of test
```

## Test Coverage Goals

We aim for:
- **80%+** overall code coverage
- **90%+** coverage for core business logic
- **100%** coverage for critical paths (authentication, data processing, etc.)

## Running Tests in CI/CD

Tests run automatically on:
- Every push to main branch
- Every pull request
- Scheduled nightly runs

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=agentic_chain --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Test Best Practices

### 1. Use Descriptive Test Names

```python
# Good
def test_analyzer_detects_python_files_correctly(self):
    ...

# Bad
def test_1(self):
    ...
```

### 2. Follow Arrange-Act-Assert Pattern

```python
def test_feature(self):
    # Arrange - set up test data
    context = AgentContext(project_path="/test")
    
    # Act - perform the action
    result = agent.execute(context)
    
    # Assert - verify the outcome
    assert result is not None
```

### 3. Test One Thing Per Test

```python
# Good - tests one specific behavior
def test_finds_python_files(self):
    assert analyzer.find_files(".py") == ["main.py"]

def test_finds_javascript_files(self):
    assert analyzer.find_files(".js") == ["app.js"]

# Bad - tests multiple things
def test_finds_all_files(self):
    assert analyzer.find_files(".py") == ["main.py"]
    assert analyzer.find_files(".js") == ["app.js"]
    assert analyzer.find_files(".go") == ["main.go"]
```

### 4. Use Fixtures for Common Setup

```python
@pytest.fixture
def configured_agent(mock_llm_config):
    """Create a configured agent."""
    return SolutionImplementer(llm_provider=mock_llm_config)

def test_with_fixture(configured_agent):
    result = configured_agent.execute(context)
    assert result is not None
```

### 5. Mock External Dependencies

Always mock:
- API calls (OpenAI, Anthropic, etc.)
- File system operations (when possible)
- Network requests
- Database queries

### 6. Test Error Cases

```python
def test_handles_missing_file(self):
    with pytest.raises(FileNotFoundError):
        analyzer.read_file("/nonexistent/file")

def test_handles_invalid_input(self):
    with pytest.raises(ValueError, match="Invalid input"):
        processor.process(None)
```

## Troubleshooting

### Tests Failing Due to Missing Dependencies

```bash
pip install -r requirements.txt
```

### Tests Failing Due to Import Errors

Make sure the package is installed in development mode:

```bash
pip install -e .
```

### Slow Tests

Use pytest markers to skip slow tests during development:

```python
@pytest.mark.slow
def test_expensive_operation():
    ...

# Run without slow tests
pytest -m "not slow"
```

### Debugging Test Failures

```bash
# Run with more verbose output
pytest -vv

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Use ipdb for better debugging
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

## Contributing Tests

When contributing code, please:

1. **Write tests for new features**
2. **Update existing tests when modifying code**
3. **Maintain or improve coverage** (don't reduce coverage)
4. **Run tests locally** before submitting PR
5. **Add docstrings** to test classes and functions

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing with pytest Book](https://pragprog.com/titles/bopytest/)
- [Effective Python Testing](https://realpython.com/pytest-python-testing/)
