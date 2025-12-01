"""
GitHub Integration Configuration.

This module handles loading and parsing configuration from
`.github/agentic-chain.yml` files.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import yaml


@dataclass
class AnalysisConfig:
    """Configuration for issue analysis."""

    enabled: bool = True
    auto_comment: bool = True
    auto_label: bool = True
    priority_threshold: int = 50  # Minimum priority score for auto-action


@dataclass
class MentionConfig:
    """Configuration for @mention triggers."""

    enabled: bool = True
    trigger: str = "@agentic-chain"
    commands: List[str] = field(
        default_factory=lambda: ["analyze", "help", "label"]
    )


@dataclass
class LabelConfig:
    """Configuration for auto-labeling."""

    enabled: bool = True
    prefix: str = ""
    max_labels: int = 5
    # Mapping from suggested labels to actual repo labels
    label_mapping: Dict[str, str] = field(default_factory=dict)


@dataclass
class GitHubConfig:
    """
    GitHub Integration configuration loaded from .github/agentic-chain.yml.

    Example configuration file:
    ```yaml
    enabled: true
    analysis:
      enabled: true
      auto_comment: true
      auto_label: true
      priority_threshold: 50
    mention:
      enabled: true
      trigger: "@agentic-chain"
      commands:
        - analyze
        - help
        - label
    labels:
      enabled: true
      prefix: "agentic:"
      max_labels: 5
      mapping:
        bug: "type:bug"
        enhancement: "type:feature"
    llm:
      provider: "openai"
      model: "gpt-4"
    ```
    """

    enabled: bool = True
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    mention: MentionConfig = field(default_factory=MentionConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "GitHubConfig":
        """
        Create a GitHubConfig from YAML content.

        Args:
            yaml_content: The YAML configuration content.

        Returns:
            Configured GitHubConfig instance.
        """
        try:
            data = yaml.safe_load(yaml_content)
            if data is None:
                data = {}
        except yaml.YAMLError:
            # Invalid YAML, return defaults
            return cls()

        return cls._from_dict(data)

    @classmethod
    def from_file(cls, file_path: str) -> "GitHubConfig":
        """
        Load configuration from a file.

        Args:
            file_path: Path to the YAML configuration file.

        Returns:
            Configured GitHubConfig instance.
        """
        if not os.path.exists(file_path):
            return cls()

        with open(file_path, "r") as f:
            return cls.from_yaml(f.read())

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "GitHubConfig":
        """Create GitHubConfig from a dictionary."""
        analysis_data = data.get("analysis", {})
        mention_data = data.get("mention", {})
        labels_data = data.get("labels", {})
        llm_data = data.get("llm", {})

        return cls(
            enabled=data.get("enabled", True),
            analysis=AnalysisConfig(
                enabled=analysis_data.get("enabled", True),
                auto_comment=analysis_data.get("auto_comment", True),
                auto_label=analysis_data.get("auto_label", True),
                priority_threshold=analysis_data.get("priority_threshold", 50),
            ),
            mention=MentionConfig(
                enabled=mention_data.get("enabled", True),
                trigger=mention_data.get("trigger", "@agentic-chain"),
                commands=mention_data.get(
                    "commands", ["analyze", "help", "label"]
                ),
            ),
            labels=LabelConfig(
                enabled=labels_data.get("enabled", True),
                prefix=labels_data.get("prefix", ""),
                max_labels=labels_data.get("max_labels", 5),
                label_mapping=labels_data.get("mapping", {}),
            ),
            llm_provider=llm_data.get("provider"),
            llm_model=llm_data.get("model"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "enabled": self.enabled,
            "analysis": {
                "enabled": self.analysis.enabled,
                "auto_comment": self.analysis.auto_comment,
                "auto_label": self.analysis.auto_label,
                "priority_threshold": self.analysis.priority_threshold,
            },
            "mention": {
                "enabled": self.mention.enabled,
                "trigger": self.mention.trigger,
                "commands": self.mention.commands,
            },
            "labels": {
                "enabled": self.labels.enabled,
                "prefix": self.labels.prefix,
                "max_labels": self.labels.max_labels,
                "mapping": self.labels.label_mapping,
            },
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
            },
        }
