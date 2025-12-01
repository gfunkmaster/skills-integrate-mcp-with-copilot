"""
Webhook Handler for GitHub Integration.

This module handles incoming GitHub webhook events, specifically
for issue-related events.
"""

import hashlib
import hmac
import logging
from typing import Dict, Any, Optional, Callable, List

from .config import GitHubConfig
from .issue_processor import IssueProcessor

logger = logging.getLogger(__name__)


class WebhookHandler:
    """
    Handles GitHub webhook events for issue processing.

    This handler supports:
    - Issue opened events (auto-analysis)
    - Issue comment events (mention commands)
    - Webhook signature verification
    """

    SUPPORTED_EVENTS = ["issues", "issue_comment"]

    def __init__(
        self,
        config: Optional[GitHubConfig] = None,
        project_path: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        """
        Initialize the WebhookHandler.

        Args:
            config: GitHub integration configuration.
            project_path: Path to the project being analyzed.
            webhook_secret: Secret for verifying webhook signatures.
        """
        self.config = config or GitHubConfig()
        self.project_path = project_path or "."
        self.webhook_secret = webhook_secret
        self.processor = IssueProcessor(
            config=self.config,
            project_path=self.project_path,
        )

        # Callbacks for GitHub API interactions
        self._post_comment_callback: Optional[Callable] = None
        self._add_labels_callback: Optional[Callable] = None

    def set_post_comment_callback(
        self,
        callback: Callable[[int, str], None],
    ) -> None:
        """
        Set the callback for posting comments to GitHub.

        Args:
            callback: Function that takes (issue_number, comment_body)
        """
        self._post_comment_callback = callback

    def set_add_labels_callback(
        self,
        callback: Callable[[int, List[str]], None],
    ) -> None:
        """
        Set the callback for adding labels to issues.

        Args:
            callback: Function that takes (issue_number, labels)
        """
        self._add_labels_callback = callback

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """
        Verify the GitHub webhook signature.

        Args:
            payload: The raw request body.
            signature: The X-Hub-Signature-256 header value.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if not self.webhook_secret:
            # No secret configured, skip verification
            return True

        if not signature:
            return False

        # Remove the "sha256=" prefix
        if signature.startswith("sha256="):
            signature = signature[7:]

        expected = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def handle_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle a GitHub webhook event.

        Args:
            event_type: The GitHub event type (X-GitHub-Event header).
            payload: The webhook payload.

        Returns:
            Dictionary containing the handling result.
        """
        if event_type not in self.SUPPORTED_EVENTS:
            return {
                "handled": False,
                "reason": f"Unsupported event type: {event_type}",
            }

        if event_type == "issues":
            return self._handle_issue_event(payload)
        elif event_type == "issue_comment":
            return self._handle_comment_event(payload)

        return {
            "handled": False,
            "reason": "Event type not implemented",
        }

    def _handle_issue_event(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle an issues webhook event."""
        action = payload.get("action")

        if action != "opened":
            return {
                "handled": False,
                "reason": f"Issue action '{action}' not handled",
            }

        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        if not self.processor.should_process_issue(issue):
            return {
                "handled": False,
                "reason": "Issue should not be processed",
            }

        # Convert issue to expected format
        issue_data = self._convert_issue_payload(issue)

        # Process the issue
        result = self.processor.process_issue(issue_data)

        if result.get("skipped"):
            return {
                "handled": False,
                "reason": result.get("reason"),
            }

        # Post comment if configured
        if result.get("comment") and self._post_comment_callback:
            try:
                self._post_comment_callback(issue_number, result["comment"])
            except Exception as e:
                logger.error(f"Failed to post comment: {e}")

        # Add labels if configured
        if (
            result.get("suggested_labels")
            and self.config.analysis.auto_label
            and self._add_labels_callback
        ):
            try:
                labels = result["suggested_labels"]
                # Add the analyzed label
                labels.append(self.processor.get_analysis_label())
                self._add_labels_callback(issue_number, labels)
            except Exception as e:
                logger.error(f"Failed to add labels: {e}")

        return {
            "handled": True,
            "issue_number": issue_number,
            "comment_posted": result.get("comment") is not None,
            "labels_added": result.get("suggested_labels", []),
        }

    def _handle_comment_event(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle an issue_comment webhook event."""
        action = payload.get("action")

        if action != "created":
            return {
                "handled": False,
                "reason": f"Comment action '{action}' not handled",
            }

        comment = payload.get("comment", {})
        comment_body = comment.get("body", "")
        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        # Check for mention command
        command = self.processor.parse_mention_command(comment_body)

        if not command:
            return {
                "handled": False,
                "reason": "No mention command found",
            }

        # Convert issue to expected format
        issue_data = self._convert_issue_payload(issue)

        # Process the command
        result = self.processor.process_mention_command(command, issue_data)

        if result.get("skipped"):
            return {
                "handled": False,
                "reason": result.get("reason"),
            }

        # Post response comment
        if result.get("comment") and self._post_comment_callback:
            try:
                self._post_comment_callback(issue_number, result["comment"])
            except Exception as e:
                logger.error(f"Failed to post comment: {e}")

        # Add labels if applicable
        if (
            result.get("suggested_labels")
            and self._add_labels_callback
            and command == "label"
        ):
            try:
                self._add_labels_callback(issue_number, result["suggested_labels"])
            except Exception as e:
                logger.error(f"Failed to add labels: {e}")

        return {
            "handled": True,
            "issue_number": issue_number,
            "command": command,
            "comment_posted": result.get("comment") is not None,
        }

    def _convert_issue_payload(
        self,
        issue: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert a GitHub issue payload to the format expected by IssueProcessor.

        Args:
            issue: The issue object from the webhook payload.

        Returns:
            Dictionary in the expected format.
        """
        return {
            "number": issue.get("number"),
            "title": issue.get("title", ""),
            "body": issue.get("body") or "",
            "labels": issue.get("labels", []),
            "state": issue.get("state"),
            "user": issue.get("user", {}).get("login"),
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
        }
