"""Performance benchmark tests for the Agentic Chain.

These tests measure execution time for key operations to ensure
the system meets performance targets (< 5 seconds for analysis).

Run with:
    pytest tests/test_benchmarks.py -v

To run benchmarks only (if pytest-benchmark is installed):
    pytest tests/test_benchmarks.py -v --benchmark-only
"""

import time
import pytest
from pathlib import Path

from agentic_chain import AgenticChain
from agentic_chain.agents import AgentContext
from agentic_chain.agents.project_analyzer import ProjectAnalyzer
from agentic_chain.agents.issue_analyzer import IssueAnalyzer
from agentic_chain.agents.code_reviewer import CodeReviewer
from agentic_chain.agents.solution_implementer import SolutionImplementer
from agentic_chain.parallel import ExecutionMode, ParallelExecutionConfig


# Performance constants (in seconds)
PROJECT_ANALYSIS_TARGET = 5.0  # Target time for project analysis
ISSUE_ANALYSIS_TARGET = 2.0   # Target time for issue analysis
CODE_REVIEW_TARGET = 3.0      # Target time for code review
SOLUTION_TARGET = 2.0         # Target time for solution generation
FULL_CHAIN_TARGET = 10.0      # Target time for full chain execution
SCALING_BASE_TIME = 0.5       # Base time overhead for analysis
SCALING_PER_FILE = 0.1        # Expected time per file for scaling tests


@pytest.fixture
def large_project(tmp_path):
    """Create a larger project structure for benchmarking."""
    # Create directories
    src = tmp_path / "src"
    src.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "docs").mkdir()
    
    # Create multiple Python files
    for i in range(10):
        (src / f"module_{i}.py").write_text(f"""
'''Module {i} documentation.'''

import os
from pathlib import Path


def function_{i}_a(param1, param2):
    '''Function A in module {i}.'''
    result = param1 + param2
    return result


def function_{i}_b(items):
    '''Function B in module {i}.'''
    total = 0
    for item in items:
        total += item
    return total


class Class{i}:
    '''Class in module {i}.'''
    
    def __init__(self, value):
        self.value = value
    
    def method_a(self):
        '''Method A.'''
        return self.value * 2
    
    def method_b(self, other):
        '''Method B.'''
        return self.value + other


# TODO: Add more functionality
# FIXME: Fix edge cases
""")
        
        # Create corresponding test file
        (tests_dir / f"test_module_{i}.py").write_text(f"""
import pytest
from src.module_{i} import function_{i}_a, function_{i}_b, Class{i}


def test_function_a():
    assert function_{i}_a(1, 2) == 3


def test_function_b():
    assert function_{i}_b([1, 2, 3]) == 6


def test_class():
    obj = Class{i}(5)
    assert obj.method_a() == 10
""")
    
    # Create config files
    (tmp_path / "requirements.txt").write_text("""
fastapi
uvicorn
pydantic
pytest
pytest-cov
""")
    
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "benchmark-project"
version = "1.0.0"
description = "A project for benchmarking"

[tool.pytest.ini_options]
testpaths = ["tests"]
""")
    
    (tmp_path / "README.md").write_text("""
# Benchmark Project

A project used for performance benchmarking of Agentic Chain.

## Features
- Multiple modules
- Full test suite
- Documentation
""")
    
    # Create GitHub Actions workflow
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: CI\non: [push]")
    
    return tmp_path


@pytest.fixture
def complex_issue():
    """Create a complex issue for benchmarking."""
    return {
        "title": "Refactor authentication module for better security",
        "body": """
## Problem
The current authentication module has several issues that need to be addressed:
1. Passwords are not properly hashed
2. Session tokens expire too quickly
3. Rate limiting is not implemented

## Proposed Solution
We need to:
- Use bcrypt for password hashing
- Extend session token expiry to 24 hours
- Implement rate limiting with Redis

## Affected Files
- `src/module_0.py`
- `src/module_1.py`
- `tests/test_module_0.py`

## Additional Context
This is a critical security issue that affects production users.
Priority: High
Estimated effort: 3-5 days

## Acceptance Criteria
1. All passwords are hashed with bcrypt
2. Session tokens have configurable expiry
3. Rate limiting prevents > 100 requests/minute per IP
""",
        "labels": [
            {"name": "bug"},
            {"name": "security"},
            {"name": "high-priority"},
            {"name": "backend"}
        ],
        "number": 42,
        "user": {"login": "testuser"},
    }


class TestAnalysisPerformance:
    """Performance tests for analysis operations."""
    
    def test_project_analysis_under_target(self, large_project):
        """Test that project analysis completes in under 5 seconds."""
        analyzer = ProjectAnalyzer()
        context = AgentContext(project_path=str(large_project))
        
        start_time = time.perf_counter()
        result = analyzer.execute(context)
        duration = time.perf_counter() - start_time
        
        assert result.project_analysis is not None
        assert duration < PROJECT_ANALYSIS_TARGET, f"Project analysis took {duration:.2f}s, target is <{PROJECT_ANALYSIS_TARGET}s"
    
    def test_issue_analysis_under_target(self, large_project, complex_issue):
        """Test that issue analysis completes in under 2 seconds."""
        analyzer = IssueAnalyzer()
        context = AgentContext(project_path=str(large_project))
        context.issue_data = complex_issue
        context.project_analysis = {
            "languages": {"Python": 10},
            "patterns": {"framework": "FastAPI"},
        }
        
        start_time = time.perf_counter()
        result = analyzer.execute(context)
        duration = time.perf_counter() - start_time
        
        assert result.issue_analysis is not None
        assert duration < ISSUE_ANALYSIS_TARGET, f"Issue analysis took {duration:.2f}s, target is <{ISSUE_ANALYSIS_TARGET}s"
    
    def test_code_review_under_target(self, large_project, complex_issue):
        """Test that code review completes in under 3 seconds."""
        reviewer = CodeReviewer()
        context = AgentContext(project_path=str(large_project))
        context.issue_data = complex_issue
        context.project_analysis = {"languages": {"Python": 10}}
        context.issue_analysis = {
            "affected_files": ["src/module_0.py", "src/module_1.py"],
            "keywords": ["authentication", "security"],
            "issue_type": "bug",
        }
        
        start_time = time.perf_counter()
        result = reviewer.execute(context)
        duration = time.perf_counter() - start_time
        
        assert result.code_review is not None
        assert duration < CODE_REVIEW_TARGET, f"Code review took {duration:.2f}s, target is <{CODE_REVIEW_TARGET}s"
    
    def test_solution_generation_under_target(self, large_project, complex_issue):
        """Test that solution generation completes in under 2 seconds."""
        implementer = SolutionImplementer()
        context = AgentContext(project_path=str(large_project))
        context.issue_data = complex_issue
        context.project_analysis = {"languages": {"Python": 10}}
        context.issue_analysis = {
            "issue_type": "bug",
            "requirements": ["Use bcrypt", "Add rate limiting"],
            "keywords": ["security", "authentication"],
        }
        context.code_review = {
            "relevant_files": ["src/module_0.py"],
            "code_quality": {"has_tests": True},
            "suggestions": [],
        }
        
        start_time = time.perf_counter()
        result = implementer.execute(context)
        duration = time.perf_counter() - start_time
        
        assert result.solution is not None
        assert duration < SOLUTION_TARGET, f"Solution generation took {duration:.2f}s, target is <{SOLUTION_TARGET}s"


class TestChainPerformance:
    """Performance tests for the full agent chain."""
    
    def test_full_chain_under_target(self, large_project, complex_issue):
        """Test that the full chain completes in under 10 seconds."""
        chain = AgenticChain(project_path=str(large_project))
        
        start_time = time.perf_counter()
        result = chain.solve_issue(complex_issue)
        duration = time.perf_counter() - start_time
        
        assert result is not None
        assert "solution" in result
        assert duration < FULL_CHAIN_TARGET, f"Full chain took {duration:.2f}s, target is <{FULL_CHAIN_TARGET}s"
    
    def test_sequential_vs_parallel_performance(self, large_project, complex_issue):
        """Compare sequential vs parallel execution performance."""
        # Sequential execution
        chain_seq = AgenticChain(
            project_path=str(large_project),
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        
        start_time = time.perf_counter()
        result_seq = chain_seq.solve_issue(complex_issue)
        duration_seq = time.perf_counter() - start_time
        
        # Parallel execution
        chain_par = AgenticChain(
            project_path=str(large_project),
            execution_mode=ExecutionMode.PARALLEL,
            parallel_config=ParallelExecutionConfig(max_workers=4),
        )
        
        start_time = time.perf_counter()
        result_par = chain_par.solve_issue(complex_issue)
        duration_par = time.perf_counter() - start_time
        
        assert result_seq is not None
        assert result_par is not None
        
        # Log performance comparison
        print(f"\nSequential: {duration_seq:.3f}s")
        print(f"Parallel: {duration_par:.3f}s")
        print(f"Speedup: {duration_seq/duration_par:.2f}x")


class TestScalabilityBenchmarks:
    """Scalability tests for different project sizes."""
    
    @pytest.mark.parametrize("num_files", [5, 10, 20])
    def test_project_analysis_scaling(self, tmp_path, num_files):
        """Test project analysis scales linearly with file count."""
        # Create project with variable number of files
        src = tmp_path / "src"
        src.mkdir()
        
        for i in range(num_files):
            (src / f"module_{i}.py").write_text(f"def func_{i}(): pass\n" * 10)
        
        analyzer = ProjectAnalyzer()
        context = AgentContext(project_path=str(tmp_path))
        
        start_time = time.perf_counter()
        result = analyzer.execute(context)
        duration = time.perf_counter() - start_time
        
        assert result.project_analysis is not None
        
        # Expect roughly linear scaling based on defined constants
        expected_max = SCALING_BASE_TIME + (num_files * SCALING_PER_FILE)
        assert duration < expected_max, f"{num_files} files took {duration:.2f}s, expected <{expected_max}s"


class TestMemoryBenchmarks:
    """Memory usage benchmarks."""
    
    def test_chain_memory_stability(self, large_project, complex_issue):
        """Test that memory usage is stable across multiple runs."""
        chain = AgenticChain(project_path=str(large_project))
        
        # Run multiple times to check for memory leaks
        for _ in range(3):
            result = chain.solve_issue(complex_issue)
            assert result is not None
        
        # If we get here without OOM, the test passes


class TestTimingBaselines:
    """Baseline timing tests to track performance regressions."""
    
    def test_baseline_project_analysis(self, large_project):
        """Baseline test for project analysis timing."""
        analyzer = ProjectAnalyzer()
        context = AgentContext(project_path=str(large_project))
        
        times = []
        for _ in range(3):
            start = time.perf_counter()
            analyzer.execute(context)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nProject Analysis Timing:")
        print(f"  Min: {min_time:.4f}s")
        print(f"  Max: {max_time:.4f}s")
        print(f"  Avg: {avg_time:.4f}s")
        
        # Ensure consistent performance (max < 2x min)
        assert max_time < min_time * 2, "High variance in timing"
    
    def test_baseline_full_chain(self, large_project, complex_issue):
        """Baseline test for full chain timing."""
        times = []
        for _ in range(3):
            chain = AgenticChain(project_path=str(large_project))
            start = time.perf_counter()
            chain.solve_issue(complex_issue)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        
        print(f"\nFull Chain Timing:")
        print(f"  Min: {min(times):.4f}s")
        print(f"  Max: {max(times):.4f}s")
        print(f"  Avg: {avg_time:.4f}s")
        
        # Full chain should complete in under 5 seconds on average
        assert avg_time < 5.0, f"Average chain time {avg_time:.2f}s exceeds 5s target"
