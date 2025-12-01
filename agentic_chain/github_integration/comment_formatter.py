"""
Comment Formatter for GitHub Integration.

This module formats analysis results into well-structured GitHub comments
using markdown formatting.
"""

from typing import Dict, Any, List, Optional


class CommentFormatter:
    """
    Formats analysis results into GitHub-compatible markdown comments.

    The formatter creates structured, readable comments that include:
    - Issue classification and priority
    - Sentiment analysis results
    - Suggested labels
    - Requirements and acceptance criteria
    - Code review findings
    - Proposed solutions
    """

    HEADER = "## 🤖 Agentic Chain Analysis"

    PRIORITY_EMOJI = {
        "critical": "🚨",
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }

    ISSUE_TYPE_EMOJI = {
        "bug": "🐛",
        "feature": "✨",
        "enhancement": "💡",
        "documentation": "📚",
        "security": "🔒",
        "performance": "⚡",
        "unknown": "❓",
    }

    def format_analysis_comment(
        self,
        analysis_result: Dict[str, Any],
        include_solution: bool = True,
    ) -> str:
        """
        Format the complete analysis result as a GitHub comment.

        Args:
            analysis_result: The result from AgenticChain.solve_issue()
            include_solution: Whether to include the solution section

        Returns:
            Formatted markdown string for the GitHub comment.
        """
        sections = [self.HEADER, ""]

        # Issue Analysis section
        if analysis_result.get("issue_analysis"):
            sections.append(
                self._format_issue_analysis(analysis_result["issue_analysis"])
            )

        # Sentiment section
        if analysis_result.get("issue_analysis", {}).get("sentiment"):
            sections.append(
                self._format_sentiment(
                    analysis_result["issue_analysis"]["sentiment"]
                )
            )

        # Suggested labels section
        if analysis_result.get("issue_analysis", {}).get("suggested_labels"):
            sections.append(
                self._format_suggested_labels(
                    analysis_result["issue_analysis"]["suggested_labels"]
                )
            )

        # Code Review section
        if analysis_result.get("code_review"):
            sections.append(
                self._format_code_review(analysis_result["code_review"])
            )

        # Solution section
        if include_solution and analysis_result.get("solution"):
            sections.append(
                self._format_solution(analysis_result["solution"])
            )

        # Footer
        sections.append(self._format_footer())

        return "\n".join(sections)

    def format_quick_analysis(
        self, issue_analysis: Dict[str, Any]
    ) -> str:
        """
        Format a quick analysis comment (just issue analysis, no solution).

        Args:
            issue_analysis: The issue analysis result.

        Returns:
            Formatted markdown string.
        """
        sections = [self.HEADER, ""]
        sections.append(self._format_issue_analysis(issue_analysis))

        if issue_analysis.get("sentiment"):
            sections.append(self._format_sentiment(issue_analysis["sentiment"]))

        if issue_analysis.get("suggested_labels"):
            sections.append(
                self._format_suggested_labels(issue_analysis["suggested_labels"])
            )

        sections.append(self._format_footer())
        return "\n".join(sections)

    def format_labels_comment(
        self, suggested_labels: List[str], applied_labels: Optional[List[str]] = None
    ) -> str:
        """
        Format a comment about label suggestions.

        Args:
            suggested_labels: Labels suggested by analysis.
            applied_labels: Labels that were actually applied (if any).

        Returns:
            Formatted markdown string.
        """
        sections = [self.HEADER, ""]
        sections.append("### 🏷️ Label Suggestions\n")

        if applied_labels:
            sections.append("**Applied labels:**")
            for label in applied_labels:
                sections.append(f"- `{label}`")
            sections.append("")

        if suggested_labels:
            remaining = [l for l in suggested_labels if l not in (applied_labels or [])]
            if remaining:
                sections.append("**Additional suggested labels:**")
                for label in remaining:
                    sections.append(f"- `{label}`")
                sections.append("")

        sections.append(self._format_footer())
        return "\n".join(sections)

    def format_help_comment(self) -> str:
        """
        Format a help message explaining available commands.

        Returns:
            Formatted markdown string with usage instructions.
        """
        return f"""{self.HEADER}

### 📖 Available Commands

You can interact with me by mentioning `@agentic-chain` followed by a command:

| Command | Description |
|---------|-------------|
| `@agentic-chain analyze` | Perform a full analysis of this issue |
| `@agentic-chain label` | Suggest labels for this issue |
| `@agentic-chain help` | Show this help message |

### 🔧 Configuration

Configure Agentic Chain by creating `.github/agentic-chain.yml` in your repository:

```yaml
enabled: true
analysis:
  auto_comment: true
  auto_label: true
mention:
  trigger: "@agentic-chain"
labels:
  max_labels: 5
```

{self._format_footer()}
"""

    def _format_issue_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format the issue analysis section."""
        issue_type = analysis.get("issue_type", "unknown")
        priority = analysis.get("priority", "medium")
        priority_score = analysis.get("priority_score", 50)

        type_emoji = self.ISSUE_TYPE_EMOJI.get(issue_type, "❓")
        priority_emoji = self.PRIORITY_EMOJI.get(priority, "🟡")

        lines = [
            "### 📋 Issue Classification\n",
            f"| Attribute | Value |",
            f"|-----------|-------|",
            f"| Type | {type_emoji} {issue_type.title()} |",
            f"| Priority | {priority_emoji} {priority.title()} (score: {priority_score}/100) |",
        ]

        if analysis.get("keywords"):
            keywords = ", ".join(analysis["keywords"][:5])
            lines.append(f"| Keywords | {keywords} |")

        lines.append("")
        return "\n".join(lines)

    def _format_sentiment(self, sentiment: Dict[str, Any]) -> str:
        """Format the sentiment analysis section."""
        urgency = sentiment.get("urgency_level", "low")
        tone = sentiment.get("tone", "neutral")
        indicators = sentiment.get("urgency_indicators", [])

        urgency_emoji = {"high": "🔥", "medium": "⚠️", "low": "✅"}.get(
            urgency, "✅"
        )

        lines = [
            "### 💭 Sentiment Analysis\n",
            f"- **Urgency Level:** {urgency_emoji} {urgency.title()}",
            f"- **Tone:** {tone.title()}",
        ]

        if indicators:
            lines.append(f"- **Urgency Indicators:** {', '.join(indicators[:3])}")

        if sentiment.get("frustration_detected"):
            lines.append("- ⚠️ **User frustration detected**")

        lines.append("")
        return "\n".join(lines)

    def _format_suggested_labels(self, labels: List[str]) -> str:
        """Format the suggested labels section."""
        if not labels:
            return ""

        label_badges = " ".join([f"`{label}`" for label in labels[:8]])

        return f"""### 🏷️ Suggested Labels

{label_badges}

"""

    def _format_code_review(self, review: Dict[str, Any]) -> str:
        """Format the code review section."""
        lines = ["### 🔍 Code Review\n"]

        relevant_files = review.get("relevant_files", [])
        if relevant_files:
            lines.append("**Relevant Files:**")
            for file_path in relevant_files[:5]:
                lines.append(f"- `{file_path}`")
            lines.append("")

        potential_issues = review.get("potential_issues", [])
        if potential_issues:
            lines.append("**Potential Issues Found:**")
            for issue in potential_issues[:3]:
                lines.append(f"- {issue.get('description', str(issue))}")
            lines.append("")

        if not relevant_files and not potential_issues:
            lines.append("_No specific code areas identified._\n")

        return "\n".join(lines)

    def _format_solution(self, solution: Dict[str, Any]) -> str:
        """Format the solution section."""
        lines = ["### 💡 Proposed Solution\n"]

        if solution.get("llm_generated"):
            lines.append("_✨ AI-generated solution_\n")

        plan = solution.get("implementation_plan", {})
        if plan:
            complexity = plan.get("complexity", "unknown")
            hours = plan.get("estimated_hours", "N/A")
            lines.append(f"**Complexity:** {complexity.title()}")
            lines.append(f"**Estimated Time:** {hours} hours\n")

        changes = solution.get("proposed_changes", [])
        if changes:
            lines.append("**Proposed Changes:**")
            for change in changes[:5]:
                change_type = change.get("type", "change")
                desc = change.get("description", str(change))[:100]
                lines.append(f"- [{change_type}] {desc}")
            lines.append("")

        risks = solution.get("risks", [])
        if risks:
            lines.append("**Risks:**")
            for risk in risks[:3]:
                level = risk.get("level", "medium")
                desc = risk.get("description", str(risk))[:80]
                lines.append(f"- ⚠️ [{level}] {desc}")
            lines.append("")

        return "\n".join(lines)

    def _format_footer(self) -> str:
        """Format the comment footer."""
        return (
            "---\n"
            "_Analysis by [Agentic Chain](https://github.com/skills/integrate-mcp-with-copilot) "
            "| Reply with `@agentic-chain help` for more options_"
        )
