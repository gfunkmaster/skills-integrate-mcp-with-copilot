"""Tests for the CLI module."""

import json
import os
import sys
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentic_chain.cli import (
    main,
    setup_logging,
    handle_analyze,
    handle_solve,
    handle_providers,
    handle_memory,
    get_llm_config,
)


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging(verbose=False)
        # No assertion needed, just verify it doesn't raise
    
    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        setup_logging(verbose=True)
        # No assertion needed, just verify it doesn't raise


class TestGetLLMConfig:
    """Tests for get_llm_config function."""
    
    def test_get_llm_config_no_llm(self):
        """Test with no LLM provider specified."""
        args = MagicMock()
        args.llm = None
        
        result = get_llm_config(args)
        
        assert result is None
    
    def test_get_llm_config_with_provider(self):
        """Test with LLM provider specified."""
        args = MagicMock()
        args.llm = "openai"
        args.model = None
        args.api_key = None
        
        result = get_llm_config(args)
        
        assert result == {"provider": "openai"}
    
    def test_get_llm_config_with_model(self):
        """Test with model specified."""
        args = MagicMock()
        args.llm = "openai"
        args.model = "gpt-4"
        args.api_key = None
        
        result = get_llm_config(args)
        
        assert result == {"provider": "openai", "model": "gpt-4"}
    
    def test_get_llm_config_with_api_key(self):
        """Test with API key specified."""
        args = MagicMock()
        args.llm = "anthropic"
        args.model = None
        args.api_key = "test-key"
        
        result = get_llm_config(args)
        
        assert result == {"provider": "anthropic", "api_key": "test-key"}
    
    def test_get_llm_config_full(self):
        """Test with all options specified."""
        args = MagicMock()
        args.llm = "openai"
        args.model = "gpt-4"
        args.api_key = "test-key"
        
        result = get_llm_config(args)
        
        assert result == {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key"
        }


class TestHandleAnalyze:
    """Tests for handle_analyze function."""
    
    def test_handle_analyze_nonexistent_path(self, capsys):
        """Test analyze with nonexistent project path."""
        args = MagicMock()
        args.project_path = "/nonexistent/path"
        args.output = None
        args.verbose = False
        
        with pytest.raises(SystemExit) as exc_info:
            handle_analyze(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.out
    
    def test_handle_analyze_valid_project(self, tmp_path, capsys):
        """Test analyze with valid project path."""
        # Create a minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.output = None
        args.verbose = False
        
        handle_analyze(args)
        
        captured = capsys.readouterr()
        assert "structure" in captured.out or "languages" in captured.out
    
    def test_handle_analyze_with_output(self, tmp_path):
        """Test analyze with output file."""
        # Create a minimal project
        (tmp_path / "app.py").write_text("print('hello')")
        output_file = tmp_path / "result.json"
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.output = str(output_file)
        args.verbose = False
        
        handle_analyze(args)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert "structure" in data


class TestHandleSolve:
    """Tests for handle_solve function."""
    
    def test_handle_solve_nonexistent_path(self, capsys):
        """Test solve with nonexistent project path."""
        args = MagicMock()
        args.project_path = "/nonexistent/path"
        args.issue_file = None
        args.title = None
        args.body = None
        args.labels = None
        args.output = None
        args.verbose = False
        args.summary = False
        args.show_usage = False
        args.llm = None
        args.model = None
        args.api_key = None
        args.interactive = False
        args.show_history = False
        
        with pytest.raises(SystemExit) as exc_info:
            handle_solve(args)
        
        assert exc_info.value.code == 1
    
    def test_handle_solve_no_issue_data(self, tmp_path, capsys):
        """Test solve without issue data."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.issue_file = None
        args.title = None
        args.body = None
        args.labels = None
        args.output = None
        args.verbose = False
        args.summary = False
        args.show_usage = False
        args.llm = None
        args.model = None
        args.api_key = None
        
        with pytest.raises(SystemExit) as exc_info:
            handle_solve(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "issue-file" in captured.out.lower() or "title" in captured.out.lower()
    
    def test_handle_solve_with_title(self, tmp_path, capsys):
        """Test solve with title option."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.issue_file = None
        args.title = "Test Issue"
        args.body = "This is a test issue"
        args.labels = ["bug", "enhancement"]
        args.output = None
        args.verbose = False
        args.summary = False
        args.show_usage = False
        args.llm = None
        args.model = None
        args.api_key = None
        args.interactive = False
        args.show_history = False
        
        handle_solve(args)
        
        captured = capsys.readouterr()
        assert "project_analysis" in captured.out or "solution" in captured.out
    
    def test_handle_solve_with_issue_file(self, tmp_path, capsys):
        """Test solve with issue file."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        issue_file = tmp_path / "issue.json"
        issue_data = {
            "title": "Test Issue",
            "body": "This is a test issue",
            "labels": [{"name": "bug"}]
        }
        issue_file.write_text(json.dumps(issue_data))
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.issue_file = str(issue_file)
        args.title = None
        args.body = None
        args.labels = None
        args.output = None
        args.verbose = False
        args.summary = False
        args.show_usage = False
        args.llm = None
        args.model = None
        args.api_key = None
        args.interactive = False
        args.show_history = False
        
        handle_solve(args)
        
        captured = capsys.readouterr()
        assert "project_analysis" in captured.out or "solution" in captured.out
    
    def test_handle_solve_with_summary(self, tmp_path, capsys):
        """Test solve with summary option."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.issue_file = None
        args.title = "Test Issue"
        args.body = "Test body"
        args.labels = None
        args.output = None
        args.verbose = False
        args.summary = True
        args.show_usage = False
        args.llm = None
        args.model = None
        args.api_key = None
        args.interactive = False
        args.show_history = False
        
        handle_solve(args)
        
        captured = capsys.readouterr()
        assert "AGENTIC CHAIN" in captured.out or "SOLUTION" in captured.out


class TestHandleProviders:
    """Tests for handle_providers function."""
    
    def test_handle_providers(self, capsys):
        """Test listing providers."""
        args = MagicMock()
        
        handle_providers(args)
        
        captured = capsys.readouterr()
        assert "openai" in captured.out.lower()
        assert "anthropic" in captured.out.lower()


class TestHandleMemory:
    """Tests for handle_memory function."""
    
    def test_handle_memory_no_subcommand(self, capsys):
        """Test memory command without subcommand."""
        args = MagicMock()
        args.memory_command = None
        
        with pytest.raises(SystemExit) as exc_info:
            handle_memory(args)
        
        assert exc_info.value.code == 1
    
    def test_handle_memory_stats(self, tmp_path, capsys):
        """Test memory stats command."""
        db_path = tmp_path / "memory.db"
        
        args = MagicMock()
        args.memory_command = "stats"
        args.db_path = str(db_path)
        
        handle_memory(args)
        
        captured = capsys.readouterr()
        assert "Memory" in captured.out or "entries" in captured.out.lower()


class TestMainCLI:
    """Tests for main CLI entry point."""
    
    def test_main_no_command(self, capsys):
        """Test main with no command shows help."""
        with patch.object(sys, 'argv', ['agentic-chain']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        
        assert exc_info.value.code == 1
    
    def test_main_analyze_command(self, tmp_path):
        """Test main with analyze command."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        with patch.object(sys, 'argv', ['agentic-chain', 'analyze', str(tmp_path)]):
            main()  # Should not raise
    
    def test_main_providers_command(self, capsys):
        """Test main with providers command."""
        with patch.object(sys, 'argv', ['agentic-chain', 'providers']):
            main()
        
        captured = capsys.readouterr()
        assert "openai" in captured.out.lower()
    
    def test_main_solve_command(self, tmp_path):
        """Test main with solve command."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        with patch.object(sys, 'argv', [
            'agentic-chain', 'solve', str(tmp_path),
            '--title', 'Test', '--body', 'Test body'
        ]):
            main()  # Should not raise


class TestCLIErrorHandling:
    """Tests for CLI error handling."""
    
    def test_invalid_issue_file(self, tmp_path, capsys):
        """Test handling of invalid issue file."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json")
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.issue_file = str(invalid_file)
        args.title = None
        args.body = None
        args.labels = None
        args.output = None
        args.verbose = False
        args.summary = False
        args.llm = None
        args.model = None
        args.api_key = None
        
        with pytest.raises(SystemExit) as exc_info:
            handle_solve(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()
    
    def test_missing_issue_file(self, tmp_path, capsys):
        """Test handling of missing issue file."""
        (tmp_path / "app.py").write_text("print('hello')")
        
        args = MagicMock()
        args.project_path = str(tmp_path)
        args.issue_file = str(tmp_path / "nonexistent.json")
        args.title = None
        args.body = None
        args.labels = None
        args.output = None
        args.verbose = False
        args.summary = False
        args.llm = None
        args.model = None
        args.api_key = None
        
        with pytest.raises(SystemExit) as exc_info:
            handle_solve(args)
        
        assert exc_info.value.code == 1
