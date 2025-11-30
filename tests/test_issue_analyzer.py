"""Tests for the IssueAnalyzer agent."""

import pytest

from agentic_chain.agents import AgentContext
from agentic_chain.agents.issue_analyzer import IssueAnalyzer


class TestIssueAnalyzer:
    """Test cases for IssueAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = IssueAnalyzer()
        
    def test_init(self):
        """Test analyzer initialization."""
        assert self.analyzer.name == "IssueAnalyzer"
        
    def test_execute_with_valid_issue(self, tmp_path):
        """Test execution with a valid issue."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Bug: Login not working",
            "body": "Users cannot login when password is empty",
            "labels": [{"name": "bug"}, {"name": "high"}]
        }
        
        result = self.analyzer.execute(context)
        
        assert result.issue_analysis is not None
        assert result.issue_analysis["title"] == "Bug: Login not working"
        assert result.issue_analysis["issue_type"] == "bug"
        
    def test_execute_without_issue_data(self, tmp_path):
        """Test execution without issue data raises error."""
        context = AgentContext(project_path=str(tmp_path))
        
        with pytest.raises(ValueError, match="Issue data is required"):
            self.analyzer.execute(context)
            
    def test_classify_issue_bug(self, tmp_path):
        """Test bug classification."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Error when saving file",
            "body": "The application crashes",
            "labels": [{"name": "bug"}]
        }
        
        result = self.analyzer.execute(context)
        assert result.issue_analysis["issue_type"] == "bug"
        
    def test_classify_issue_feature(self, tmp_path):
        """Test feature classification."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Add dark mode",
            "body": "Implement a new dark mode feature",
            "labels": [{"name": "feature"}]
        }
        
        result = self.analyzer.execute(context)
        assert result.issue_analysis["issue_type"] == "feature"
        
    def test_classify_issue_enhancement(self, tmp_path):
        """Test enhancement classification."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Improve performance",
            "body": "Optimize the database queries",
            "labels": [{"name": "enhancement"}]
        }
        
        result = self.analyzer.execute(context)
        assert result.issue_analysis["issue_type"] == "enhancement"
        
    def test_determine_priority_from_labels(self, tmp_path):
        """Test priority determination from labels."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Security vulnerability",
            "body": "Critical security issue",
            "labels": [{"name": "critical"}, {"name": "security"}]
        }
        
        result = self.analyzer.execute(context)
        assert result.issue_analysis["priority"] == "critical"
        
    def test_extract_file_references(self, tmp_path):
        """Test file reference extraction."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Fix issue in auth module",
            "body": "The bug is in `src/auth/login.py` file. Also check `config.json`.",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        affected_files = result.issue_analysis["affected_files"]
        assert "src/auth/login.py" in affected_files
        assert "config.json" in affected_files
        
    def test_extract_requirements_from_bullet_list(self, tmp_path):
        """Test requirement extraction from bullet list."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Add user profile feature",
            "body": """We need to:
- Add user profile page
- Allow users to update their email
- Add profile picture upload
""",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        requirements = result.issue_analysis["requirements"]
        assert len(requirements) >= 3
        assert any("profile page" in req for req in requirements)
        
    def test_extract_requirements_from_numbered_list(self, tmp_path):
        """Test requirement extraction from numbered list."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Multi-step form",
            "body": """Steps to implement:
1. Create form component
2. Add validation
3. Submit to API
""",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        requirements = result.issue_analysis["requirements"]
        assert len(requirements) >= 3
        
    def test_extract_keywords(self, tmp_path):
        """Test keyword extraction."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Database connection issue",
            "body": "The database connection times out frequently",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        keywords = result.issue_analysis["keywords"]
        assert "database" in keywords
        assert "connection" in keywords
        
    def test_identify_components_with_project_analysis(self, tmp_path):
        """Test component identification with project analysis."""
        context = AgentContext(project_path=str(tmp_path))
        context.project_analysis = {
            "languages": {"Python": 10},
            "patterns": {"framework": "FastAPI"},
        }
        context.issue_data = {
            "title": "FastAPI endpoint issue",
            "body": "The Python endpoint returns 500",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        components = result.issue_analysis["related_components"]
        assert any("Python" in c for c in components)
        assert any("FastAPI" in c for c in components)
        
    def test_analyze_sentiment_high_urgency(self, tmp_path):
        """Test sentiment analysis with high urgency issue."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "URGENT: Production is down!",
            "body": "Critical issue - our production server is broken. Need fix immediately!",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        sentiment = result.issue_analysis["sentiment"]
        
        assert sentiment["urgency_level"] == "high"
        assert sentiment["urgency_score"] >= 4
        assert "urgent" in sentiment["urgency_indicators"] or "critical" in sentiment["urgency_indicators"]
        
    def test_analyze_sentiment_low_urgency(self, tmp_path):
        """Test sentiment analysis with low urgency issue."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Add dark mode option",
            "body": "It would be nice to have a dark mode for the application.",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        sentiment = result.issue_analysis["sentiment"]
        
        assert sentiment["urgency_level"] == "low"
        assert sentiment["tone"] == "neutral"
        
    def test_calculate_priority_score(self, tmp_path):
        """Test priority score calculation."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Critical security vulnerability",
            "body": "This is urgent and affects production",
            "labels": [{"name": "critical"}, {"name": "security"}]
        }
        
        result = self.analyzer.execute(context)
        priority_score = result.issue_analysis["priority_score"]
        
        assert priority_score >= 80  # High priority score expected
        assert priority_score <= 100
        
    def test_suggest_labels_for_bug(self, tmp_path):
        """Test label suggestions for bug issues."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Application crashes on login",
            "body": "The app crashes with an error when trying to login. This bug affects the API endpoint.",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        suggested_labels = result.issue_analysis["suggested_labels"]
        
        assert "bug" in suggested_labels
        assert "api" in suggested_labels
        
    def test_suggest_labels_for_feature(self, tmp_path):
        """Test label suggestions for feature requests."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Add new user profile page",
            "body": "We need to implement a new feature for user profiles with UI improvements.",
            "labels": []
        }
        
        result = self.analyzer.execute(context)
        suggested_labels = result.issue_analysis["suggested_labels"]
        
        assert "enhancement" in suggested_labels
        assert "frontend" in suggested_labels
