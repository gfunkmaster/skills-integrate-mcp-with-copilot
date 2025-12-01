"""
Pytest configuration and shared fixtures.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agentic_chain.llm.base import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    MessageRole,
)


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project structure for testing."""
    # Create directories
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    
    # Create Python files
    (src_dir / "__init__.py").write_text("")
    (src_dir / "main.py").write_text("""
def hello_world():
    '''Return a greeting.'''
    return "Hello, World!"

def add(a, b):
    '''Add two numbers.'''
    return a + b
""")
    
    (tests_dir / "test_main.py").write_text("""
import pytest
from src.main import hello_world, add

def test_hello_world():
    assert hello_world() == "Hello, World!"

def test_add():
    assert add(2, 3) == 5
""")
    
    # Create requirements.txt
    (tmp_path / "requirements.txt").write_text("pytest\nfastapi\nuvicorn\n")
    
    # Create README
    (tmp_path / "README.md").write_text("""
# Sample Project

A sample project for testing.

## Features
- Simple greeting function
- Add numbers
""")
    
    # Create pyproject.toml
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "sample-project"
version = "0.1.0"
description = "A sample project"
""")
    
    return tmp_path


@pytest.fixture
def sample_issue():
    """Create a sample issue for testing."""
    return {
        "title": "Add subtract function",
        "body": """
We need to add a subtract function to complement the add function.

Requirements:
- Accept two numbers as parameters
- Return the difference
- Include proper documentation
- Add unit tests

Affected files: `src/main.py`, `tests/test_main.py`
""",
        "labels": [{"name": "enhancement"}, {"name": "good first issue"}],
        "number": 123,
        "user": {"login": "testuser"},
    }


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key-123",
        temperature=0.7,
        max_tokens=1000,
        max_retries=2,
        retry_delay=0.1,
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return LLMResponse(
        content="This is a test response from the LLM.",
        model="gpt-4o-mini",
        usage=LLMUsage(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            estimated_cost=0.001,
        ),
        finish_reason="stop",
    )


@pytest.fixture
def mock_openai_client(mock_llm_response):
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    
    # Mock chat completion response
    mock_choice = MagicMock()
    mock_choice.message.content = mock_llm_response.content
    mock_choice.finish_reason = mock_llm_response.finish_reason
    
    mock_response = MagicMock()
    mock_response.model = mock_llm_response.model
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = mock_llm_response.usage.prompt_tokens
    mock_response.usage.completion_tokens = mock_llm_response.usage.completion_tokens
    mock_response.usage.total_tokens = mock_llm_response.usage.total_tokens
    mock_response.model_dump.return_value = {"model": mock_llm_response.model}
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_client(mock_llm_response):
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    
    # Mock message response
    mock_content = MagicMock()
    mock_content.text = mock_llm_response.content
    
    mock_response = MagicMock()
    mock_response.model = mock_llm_response.model
    mock_response.content = [mock_content]
    mock_response.stop_reason = mock_llm_response.finish_reason
    mock_response.usage.input_tokens = mock_llm_response.usage.prompt_tokens
    mock_response.usage.output_tokens = mock_llm_response.usage.completion_tokens
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def sample_code():
    """Sample code for testing code review."""
    return """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total

# TODO: Add error handling
# TODO: Validate input
"""


@pytest.fixture
def sample_llm_messages():
    """Sample LLM messages for testing."""
    return [
        LLMMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(role=MessageRole.USER, content="What is 2 + 2?"),
    ]
