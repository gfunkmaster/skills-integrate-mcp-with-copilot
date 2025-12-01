"""
Issue Processor for GitHub Integration.

This module handles the processing of GitHub issues, including analysis,
label suggestion, and comment generation.
"""

import logging
from typing import Dict, Any, List, Optional

from ..orchestrator import AgenticChain
from .config import GitHubConfig
from .comment_formatter import CommentFormatter

logger = logging.getLogger(__name__)


class IssueProcessor:
    """
    Processes GitHub issues using the Agentic Chain.

    This class handles:
    - Issue analysis and classification
    - Label suggestions based on analysis
    - Comment generation for posting back to GitHub
    - Mention command handling
    """

    def __init__(
        self,
        config: Optional[GitHubConfig] = None,
        project_path: Optional[str] = None,
    ):
        """
        Initialize the IssueProcessor.

        Args:
            config: GitHub integration configuration.
            project_path: Path to the project being analyzed.
        """
        self.config = config or GitHubConfig()
        self.project_path = project_path or "."
        self.formatter = CommentFormatter()

    def process_issue(
        self,
        issue_data: Dict[str, Any],
        full_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a GitHub issue and generate analysis results.

        Args:
            issue_data: Dictionary containing issue information:
                - title: Issue title
                - body: Issue description
                - labels: List of existing labels
                - number: Issue number
            full_analysis: Whether to perform full analysis or quick analysis.

        Returns:
            Dictionary containing:
                - analysis: The analysis results
                - comment: Formatted comment for GitHub
                - suggested_labels: List of suggested labels
        """
        if not self.config.enabled:
            return {
                "analysis": None,
                "comment": None,
                "suggested_labels": [],
                "skipped": True,
                "reason": "Integration is disabled",
            }

        # Prepare LLM config if available
        llm_config = None
        if self.config.llm_provider:
            llm_config = {
                "provider": self.config.llm_provider,
            }
            if self.config.llm_model:
                llm_config["model"] = self.config.llm_model

        # Create the agentic chain
        chain = AgenticChain(
            project_path=self.project_path,
            llm_config=llm_config,
        )

        try:
            # Run the analysis
            analysis_result = chain.solve_issue(issue_data)

            # Extract suggested labels
            suggested_labels = self._get_suggested_labels(analysis_result)

            # Generate comment
            if self.config.analysis.auto_comment:
                comment = self.formatter.format_analysis_comment(
                    analysis_result,
                    include_solution=full_analysis,
                )
            else:
                comment = None

            return {
                "analysis": analysis_result,
                "comment": comment,
                "suggested_labels": suggested_labels,
                "skipped": False,
            }

        except Exception as e:
            logger.error(f"Error processing issue: {e}")
            return {
                "analysis": None,
                "comment": None,
                "suggested_labels": [],
                "skipped": True,
                "reason": f"Error during analysis: {str(e)}",
            }

    def process_mention_command(
        self,
        command: str,
        issue_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a mention command from a comment.

        Args:
            command: The command to execute (e.g., "analyze", "help", "label").
            issue_data: Dictionary containing issue information.

        Returns:
            Dictionary containing the response.
        """
        if not self.config.mention.enabled:
            return {
                "comment": None,
                "skipped": True,
                "reason": "Mention commands are disabled",
            }

        command = command.lower().strip()

        if command not in self.config.mention.commands:
            return {
                "comment": None,
                "skipped": True,
                "reason": f"Unknown command: {command}",
            }

        if command == "help":
            return {
                "comment": self.formatter.format_help_comment(),
                "skipped": False,
            }

        if command == "analyze":
            result = self.process_issue(issue_data, full_analysis=True)
            return {
                "comment": result.get("comment"),
                "suggested_labels": result.get("suggested_labels", []),
                "skipped": result.get("skipped", False),
                "reason": result.get("reason"),
            }

        if command == "label":
            result = self.process_issue(issue_data, full_analysis=False)
            if result.get("skipped"):
                return result

            suggested_labels = result.get("suggested_labels", [])
            comment = self.formatter.format_labels_comment(
                suggested_labels=suggested_labels,
                applied_labels=None,  # Will be set by webhook handler
            )
            return {
                "comment": comment,
                "suggested_labels": suggested_labels,
                "skipped": False,
            }

        return {
            "comment": None,
            "skipped": True,
            "reason": f"Unhandled command: {command}",
        }

    def parse_mention_command(self, comment_body: str) -> Optional[str]:
        """
        Parse a mention command from a comment body.

        Args:
            comment_body: The body of the comment.

        Returns:
            The command if found, None otherwise.
        """
        trigger = self.config.mention.trigger.lower()
        body_lower = comment_body.lower()

        if trigger not in body_lower:
            return None

        # Find the command after the trigger
        trigger_index = body_lower.find(trigger)
        after_trigger = comment_body[trigger_index + len(trigger):].strip()

        # Get the first word after the trigger
        parts = after_trigger.split()
        if parts:
            return parts[0].lower()

        return None

    def _get_suggested_labels(
        self,
        analysis_result: Dict[str, Any],
    ) -> List[str]:
        """
        Extract and map suggested labels from analysis results.

        Args:
            analysis_result: The complete analysis result.

        Returns:
            List of suggested label names.
        """
        if not self.config.labels.enabled:
            return []

        suggested = (
            analysis_result.get("issue_analysis", {}).get("suggested_labels", [])
        )

        # Apply label mapping if configured
        mapped_labels = []
        for label in suggested:
            if label in self.config.labels.label_mapping:
                mapped_labels.append(self.config.labels.label_mapping[label])
            else:
                # Apply prefix if configured
                if self.config.labels.prefix:
                    mapped_labels.append(f"{self.config.labels.prefix}{label}")
                else:
                    mapped_labels.append(label)

        # Limit number of labels
        return mapped_labels[: self.config.labels.max_labels]

    def should_process_issue(
        self,
        issue_data: Dict[str, Any],
    ) -> bool:
        """
        Determine if an issue should be processed based on configuration.

        Args:
            issue_data: Dictionary containing issue information.

        Returns:
            True if the issue should be processed, False otherwise.
        """
        if not self.config.enabled:
            return False

        if not self.config.analysis.enabled:
            return False

        # Skip if already analyzed (has agentic-chain label or comment)
        existing_labels = [
            l.get("name", "").lower() for l in issue_data.get("labels", [])
        ]
        analysis_label = self.get_analysis_label().lower()
        if analysis_label in existing_labels:
            return False

        return True

    def get_analysis_label(self) -> str:
        """Get the label to mark issues as analyzed."""
        prefix = self.config.labels.prefix
        return f"{prefix}analyzed" if prefix else "agentic-chain-analyzed"
