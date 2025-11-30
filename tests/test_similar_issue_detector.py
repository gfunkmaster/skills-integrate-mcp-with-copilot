"""Tests for the SimilarIssueDetector agent."""

import pytest

from agentic_chain.agents import AgentContext
from agentic_chain.agents.similar_issue_detector import SimilarIssueDetector


class TestSimilarIssueDetector:
    """Test cases for SimilarIssueDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SimilarIssueDetector()
        
    def test_init(self):
        """Test detector initialization."""
        assert self.detector.name == "SimilarIssueDetector"
        
    def test_execute_without_issue_data(self, tmp_path):
        """Test execution without issue data raises error."""
        context = AgentContext(project_path=str(tmp_path))
        
        with pytest.raises(ValueError, match="Issue data is required"):
            self.detector.execute(context)
            
    def test_execute_without_existing_issues(self, tmp_path):
        """Test execution without existing issues returns empty results."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "title": "Login button not working",
            "body": "The login button fails to submit",
            "labels": []
        }
        
        result = self.detector.execute(context)
        
        assert result.metadata["similar_issues"] == []
        assert result.metadata["potential_duplicates"] == []
        
    def test_find_similar_issues(self, tmp_path):
        """Test finding similar issues from existing issues."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "number": 10,
            "title": "Login button not working",
            "body": "The login button fails to submit the form. Users cannot login.",
            "labels": []
        }
        context.metadata["existing_issues"] = [
            {
                "number": 5,
                "title": "Login button fails to submit",
                "body": "Login button not working, form fails to submit",
                "labels": []
            },
            {
                "number": 3,
                "title": "Add dark mode feature",
                "body": "We need dark mode for the UI",
                "labels": []
            },
        ]
        
        result = self.detector.execute(context)
        
        similar = result.metadata["similar_issues"]
        assert len(similar) >= 1
        # The login-related issue should be more similar
        assert similar[0]["issue_number"] == 5
        assert similar[0]["similarity_score"] > 0.3
        
    def test_detect_potential_duplicates(self, tmp_path):
        """Test detection of potential duplicate issues."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "number": 10,
            "title": "Login button not working on mobile",
            "body": "The login button fails to submit on mobile devices",
            "labels": []
        }
        context.metadata["existing_issues"] = [
            {
                "number": 5,
                "title": "Login button not working on mobile devices",
                "body": "Login button fails to submit the form on mobile",
                "labels": []
            },
        ]
        
        result = self.detector.execute(context)
        
        duplicates = result.metadata["potential_duplicates"]
        assert len(duplicates) >= 1
        assert duplicates[0]["issue_number"] == 5
        assert duplicates[0]["is_potential_duplicate"] is True
        
    def test_skip_self_comparison(self, tmp_path):
        """Test that current issue is not compared with itself."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "number": 5,
            "title": "Login button not working",
            "body": "The login button fails",
            "labels": []
        }
        context.metadata["existing_issues"] = [
            {
                "number": 5,
                "title": "Login button not working",
                "body": "The login button fails",
                "labels": []
            },
        ]
        
        result = self.detector.execute(context)
        
        # Should not find itself as a similar issue
        assert result.metadata["similar_issues"] == []
        
    def test_similarity_score_range(self, tmp_path):
        """Test that similarity scores are within valid range."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "number": 10,
            "title": "Database connection error",
            "body": "Getting connection timeout errors",
            "labels": []
        }
        context.metadata["existing_issues"] = [
            {
                "number": 1,
                "title": "Database timeout issue",
                "body": "Connection timeouts occurring",
                "labels": []
            },
            {
                "number": 2,
                "title": "Add caching layer",
                "body": "Implement Redis caching",
                "labels": []
            },
        ]
        
        result = self.detector.execute(context)
        
        for issue in result.metadata["similar_issues"]:
            assert 0 <= issue["similarity_score"] <= 1
            
    def test_matching_keywords_included(self, tmp_path):
        """Test that matching keywords are included in results."""
        context = AgentContext(project_path=str(tmp_path))
        context.issue_data = {
            "number": 10,
            "title": "Authentication error on login",
            "body": "Users getting authentication errors",
            "labels": []
        }
        context.metadata["existing_issues"] = [
            {
                "number": 5,
                "title": "Login authentication fails",
                "body": "Authentication not working for users",
                "labels": []
            },
        ]
        
        result = self.detector.execute(context)
        
        if result.metadata["similar_issues"]:
            matching = result.metadata["similar_issues"][0]["matching_keywords"]
            assert len(matching) > 0
            assert any(kw in ["authentication", "login", "users", "error"] for kw in matching)
            
    def test_deduplication_report(self, tmp_path):
        """Test generating deduplication report."""
        context = AgentContext(project_path=str(tmp_path))
        context.metadata["similar_issues"] = [
            {
                "issue_number": 5,
                "title": "Login issue",
                "similarity_score": 0.8,
                "is_potential_duplicate": True,
                "matching_keywords": ["login"],
            },
            {
                "issue_number": 3,
                "title": "Related auth issue",
                "similarity_score": 0.45,
                "is_potential_duplicate": False,
                "matching_keywords": ["auth"],
            },
        ]
        context.metadata["potential_duplicates"] = [
            context.metadata["similar_issues"][0]
        ]
        
        report = self.detector.get_deduplication_report(context)
        
        assert "Similar Issue Analysis" in report
        assert "Potential Duplicates" in report
        assert "#5" in report
        assert "Related Issues" in report
        
    def test_deduplication_report_no_similar(self, tmp_path):
        """Test deduplication report when no similar issues found."""
        context = AgentContext(project_path=str(tmp_path))
        context.metadata["similar_issues"] = []
        context.metadata["potential_duplicates"] = []
        
        report = self.detector.get_deduplication_report(context)
        
        assert "No similar or duplicate issues found" in report
