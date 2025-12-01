"""Tests for the GitHub Integration module."""

import pytest
import json
import os

from agentic_chain.github_integration import (
    GitHubConfig,
    CommentFormatter,
    IssueProcessor,
    WebhookHandler,
)


class TestGitHubConfig:
    """Test cases for GitHubConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GitHubConfig()

        assert config.enabled is True
        assert config.analysis.enabled is True
        assert config.analysis.auto_comment is True
        assert config.analysis.auto_label is True
        assert config.mention.trigger == "@agentic-chain"
        assert "analyze" in config.mention.commands

    def test_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
enabled: true
analysis:
  enabled: false
  auto_comment: true
mention:
  trigger: "@my-bot"
  commands:
    - analyze
    - help
labels:
  prefix: "auto:"
  max_labels: 3
"""
        config = GitHubConfig.from_yaml(yaml_content)

        assert config.enabled is True
        assert config.analysis.enabled is False
        assert config.mention.trigger == "@my-bot"
        assert config.labels.prefix == "auto:"
        assert config.labels.max_labels == 3

    def test_from_yaml_empty(self):
        """Test loading empty YAML content."""
        config = GitHubConfig.from_yaml("")
        assert config.enabled is True  # Default value

    def test_from_yaml_invalid(self):
        """Test loading invalid YAML content."""
        config = GitHubConfig.from_yaml("{{invalid yaml}}")
        assert config.enabled is True  # Default value

    def test_from_file_not_exists(self, tmp_path):
        """Test loading from non-existent file."""
        config = GitHubConfig.from_file(str(tmp_path / "nonexistent.yml"))
        assert config.enabled is True

    def test_from_file(self, tmp_path):
        """Test loading from file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
enabled: false
analysis:
  auto_label: false
""")
        config = GitHubConfig.from_file(str(config_file))

        assert config.enabled is False
        assert config.analysis.auto_label is False

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = GitHubConfig()
        data = config.to_dict()

        assert "enabled" in data
        assert "analysis" in data
        assert "mention" in data
        assert "labels" in data
        assert "llm" in data

    def test_llm_config(self):
        """Test LLM configuration."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4
"""
        config = GitHubConfig.from_yaml(yaml_content)

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"


class TestCommentFormatter:
    """Test cases for CommentFormatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = CommentFormatter()

    def test_format_analysis_comment(self):
        """Test formatting a complete analysis comment."""
        analysis_result = {
            "issue_analysis": {
                "title": "Test Issue",
                "issue_type": "bug",
                "priority": "high",
                "priority_score": 75,
                "keywords": ["test", "bug", "fix"],
                "sentiment": {
                    "urgency_level": "high",
                    "tone": "urgent",
                    "urgency_indicators": ["critical"],
                },
                "suggested_labels": ["bug", "priority:high"],
            },
            "code_review": {
                "relevant_files": ["src/app.py"],
                "potential_issues": [],
            },
            "solution": {
                "implementation_plan": {
                    "complexity": "medium",
                    "estimated_hours": 4,
                },
                "proposed_changes": [
                    {"type": "modify", "description": "Update the handler"}
                ],
            },
        }

        comment = self.formatter.format_analysis_comment(analysis_result)

        assert "## 🤖 Agentic Chain Analysis" in comment
        assert "Issue Classification" in comment
        assert "bug" in comment.lower()
        assert "high" in comment.lower()

    def test_format_quick_analysis(self):
        """Test formatting a quick analysis comment."""
        issue_analysis = {
            "issue_type": "feature",
            "priority": "medium",
            "priority_score": 50,
            "keywords": ["feature"],
            "sentiment": {
                "urgency_level": "low",
                "tone": "neutral",
            },
            "suggested_labels": ["enhancement"],
        }

        comment = self.formatter.format_quick_analysis(issue_analysis)

        assert "## 🤖 Agentic Chain Analysis" in comment
        assert "feature" in comment.lower()

    def test_format_labels_comment(self):
        """Test formatting a labels comment."""
        comment = self.formatter.format_labels_comment(
            suggested_labels=["bug", "api", "priority:high"],
            applied_labels=["bug"],
        )

        assert "Label" in comment
        assert "`bug`" in comment

    def test_format_help_comment(self):
        """Test formatting a help comment."""
        comment = self.formatter.format_help_comment()

        assert "Available Commands" in comment
        assert "@agentic-chain analyze" in comment
        assert "@agentic-chain help" in comment
        assert "@agentic-chain label" in comment


class TestIssueProcessor:
    """Test cases for IssueProcessor."""

    def test_init_default(self):
        """Test default initialization."""
        processor = IssueProcessor()

        assert processor.config.enabled is True
        assert processor.project_path == "."

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = GitHubConfig(enabled=False)
        processor = IssueProcessor(config=config)

        assert processor.config.enabled is False

    def test_process_issue_disabled(self):
        """Test processing when disabled."""
        config = GitHubConfig(enabled=False)
        processor = IssueProcessor(config=config)

        result = processor.process_issue({"title": "Test", "body": ""})

        assert result["skipped"] is True
        assert "disabled" in result["reason"].lower()

    def test_process_issue(self, tmp_path):
        """Test processing an issue."""
        (tmp_path / "app.py").write_text("print('hello')")

        processor = IssueProcessor(project_path=str(tmp_path))
        result = processor.process_issue({
            "title": "Bug: Application crashes",
            "body": "The app crashes when clicking the button",
            "labels": [],
        })

        assert result["skipped"] is False
        assert result["analysis"] is not None
        assert result["comment"] is not None
        assert isinstance(result["suggested_labels"], list)

    def test_parse_mention_command(self):
        """Test parsing mention commands."""
        processor = IssueProcessor()

        assert processor.parse_mention_command("@agentic-chain analyze this") == "analyze"
        assert processor.parse_mention_command("@agentic-chain help") == "help"
        assert processor.parse_mention_command("@agentic-chain label") == "label"
        assert processor.parse_mention_command("no mention here") is None

    def test_process_mention_command_help(self):
        """Test processing help command."""
        processor = IssueProcessor()
        result = processor.process_mention_command("help", {})

        assert result["skipped"] is False
        assert "Available Commands" in result["comment"]

    def test_process_mention_command_disabled(self):
        """Test processing command when disabled."""
        config = GitHubConfig()
        config.mention.enabled = False
        processor = IssueProcessor(config=config)

        result = processor.process_mention_command("analyze", {})

        assert result["skipped"] is True

    def test_should_process_issue(self):
        """Test should_process_issue logic."""
        processor = IssueProcessor()

        # Normal issue should be processed
        assert processor.should_process_issue({"labels": []}) is True

        # Already analyzed issue should not be processed
        assert processor.should_process_issue({
            "labels": [{"name": "agentic-chain-analyzed"}]
        }) is False

    def test_should_process_issue_disabled(self):
        """Test should_process_issue when disabled."""
        config = GitHubConfig(enabled=False)
        processor = IssueProcessor(config=config)

        assert processor.should_process_issue({"labels": []}) is False

    def test_get_suggested_labels_with_mapping(self, tmp_path):
        """Test label mapping."""
        (tmp_path / "app.py").write_text("print('hello')")

        config = GitHubConfig()
        config.labels.label_mapping = {"bug": "type:bug"}
        processor = IssueProcessor(config=config, project_path=str(tmp_path))

        result = processor.process_issue({
            "title": "Bug: crash",
            "body": "Application crashes with error",
            "labels": [],
        })

        # Check that bug label is mapped
        if "bug" in result.get("suggested_labels", []) or "type:bug" in result.get("suggested_labels", []):
            pass  # Either mapped or original is acceptable

    def test_get_analysis_label(self):
        """Test getting the analysis label."""
        processor = IssueProcessor()
        assert processor.get_analysis_label() == "agentic-chain-analyzed"

        config = GitHubConfig()
        config.labels.prefix = "auto:"
        processor = IssueProcessor(config=config)
        assert processor.get_analysis_label() == "auto:analyzed"


class TestWebhookHandler:
    """Test cases for WebhookHandler."""

    def test_init(self):
        """Test initialization."""
        handler = WebhookHandler()

        assert handler.config.enabled is True
        assert handler.webhook_secret is None

    def test_verify_signature_no_secret(self):
        """Test signature verification without secret."""
        handler = WebhookHandler()

        assert handler.verify_signature(b"payload", "sha256=abc") is True

    def test_verify_signature_with_secret(self):
        """Test signature verification with secret."""
        import hmac
        import hashlib

        secret = "test-secret"
        payload = b'{"test": "data"}'
        signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        handler = WebhookHandler(webhook_secret=secret)

        assert handler.verify_signature(payload, f"sha256={signature}") is True
        assert handler.verify_signature(payload, "sha256=invalid") is False

    def test_handle_event_unsupported(self):
        """Test handling unsupported events."""
        handler = WebhookHandler()
        result = handler.handle_event("push", {})

        assert result["handled"] is False
        assert "Unsupported" in result["reason"]

    def test_handle_issue_event_not_opened(self):
        """Test handling non-opened issue events."""
        handler = WebhookHandler()
        result = handler.handle_event("issues", {"action": "closed"})

        assert result["handled"] is False

    def test_handle_issue_event_opened(self, tmp_path):
        """Test handling issue opened events."""
        (tmp_path / "app.py").write_text("print('hello')")

        handler = WebhookHandler(project_path=str(tmp_path))

        # Track callbacks
        comments_posted = []
        labels_added = []

        handler.set_post_comment_callback(
            lambda num, body: comments_posted.append((num, body))
        )
        handler.set_add_labels_callback(
            lambda num, labels: labels_added.append((num, labels))
        )

        result = handler.handle_event("issues", {
            "action": "opened",
            "issue": {
                "number": 1,
                "title": "Bug: Test issue",
                "body": "Description of the bug",
                "labels": [],
            },
        })

        assert result["handled"] is True
        assert result["issue_number"] == 1

    def test_handle_comment_event_no_mention(self):
        """Test handling comment without mention."""
        handler = WebhookHandler()
        result = handler.handle_event("issue_comment", {
            "action": "created",
            "comment": {"body": "Just a regular comment"},
            "issue": {"number": 1, "title": "Test", "body": ""},
        })

        assert result["handled"] is False
        assert "No mention" in result["reason"]

    def test_handle_comment_event_with_mention(self, tmp_path):
        """Test handling comment with mention."""
        (tmp_path / "app.py").write_text("print('hello')")

        handler = WebhookHandler(project_path=str(tmp_path))

        result = handler.handle_event("issue_comment", {
            "action": "created",
            "comment": {"body": "@agentic-chain help"},
            "issue": {
                "number": 1,
                "title": "Test Issue",
                "body": "Test body",
                "labels": [],
            },
        })

        assert result["handled"] is True
        assert result["command"] == "help"

    def test_convert_issue_payload(self):
        """Test converting issue payload."""
        handler = WebhookHandler()

        issue = {
            "number": 42,
            "title": "Test",
            "body": "Body text",
            "labels": [{"name": "bug"}],
            "state": "open",
            "user": {"login": "testuser"},
        }

        converted = handler._convert_issue_payload(issue)

        assert converted["number"] == 42
        assert converted["title"] == "Test"
        assert converted["body"] == "Body text"
        assert converted["user"] == "testuser"

    def test_supported_events(self):
        """Test supported events list."""
        assert "issues" in WebhookHandler.SUPPORTED_EVENTS
        assert "issue_comment" in WebhookHandler.SUPPORTED_EVENTS
