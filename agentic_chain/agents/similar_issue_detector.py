"""
Similar Issue Detector Agent - Finds potentially related or duplicate issues.

This module provides lightweight similarity detection for GitHub issues
to help identify duplicates and related issues quickly.
"""

import re
from typing import List, Dict, Set, Tuple
from . import BaseAgent, AgentContext


class SimilarIssueDetector(BaseAgent):
    """
    Detects similar issues based on keyword matching and text similarity.
    
    This is a lightweight implementation optimized for speed (< 5 seconds)
    that doesn't require external NLP libraries.
    """
    
    def __init__(self, name: str = "SimilarIssueDetector"):
        super().__init__(name)
        
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Analyze the current issue and find similar issues if a history is provided.
        """
        if not context.issue_data:
            raise ValueError("Issue data is required for similarity detection")
        
        # Get existing issues from metadata if provided
        existing_issues = context.metadata.get("existing_issues", [])
        
        current_issue = context.issue_data
        
        # Calculate similarity with existing issues
        similar_issues = self._find_similar_issues(current_issue, existing_issues)
        
        # Store results in metadata
        context.metadata["similar_issues"] = similar_issues
        context.metadata["potential_duplicates"] = [
            issue for issue in similar_issues if issue["similarity_score"] >= 0.7
        ]
        
        return context
    
    def _find_similar_issues(
        self, 
        current_issue: dict, 
        existing_issues: List[dict]
    ) -> List[dict]:
        """
        Find issues similar to the current one.
        
        Args:
            current_issue: The issue to compare
            existing_issues: List of existing issues to compare against
            
        Returns:
            List of similar issues with similarity scores
        """
        if not existing_issues:
            return []
        
        current_keywords = self._extract_keywords(current_issue)
        current_title = current_issue.get("title", "").lower()
        
        similar = []
        
        for issue in existing_issues:
            # Skip if comparing to self
            if issue.get("number") == current_issue.get("number"):
                continue
            
            # Calculate keyword similarity
            issue_keywords = self._extract_keywords(issue)
            keyword_score = self._calculate_jaccard_similarity(
                current_keywords, issue_keywords
            )
            
            # Calculate title similarity
            issue_title = issue.get("title", "").lower()
            title_score = self._calculate_title_similarity(current_title, issue_title)
            
            # Weighted combination
            similarity_score = (keyword_score * 0.6) + (title_score * 0.4)
            
            if similarity_score >= 0.3:  # Threshold for relevance
                similar.append({
                    "issue_number": issue.get("number"),
                    "title": issue.get("title"),
                    "similarity_score": round(similarity_score, 2),
                    "matching_keywords": list(current_keywords & issue_keywords)[:5],
                    "is_potential_duplicate": similarity_score >= 0.7,
                })
        
        # Sort by similarity score, highest first
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return similar[:10]  # Return top 10 similar issues
    
    def _extract_keywords(self, issue: dict) -> Set[str]:
        """Extract meaningful keywords from an issue."""
        title = issue.get("title", "")
        body = issue.get("body", "")
        content = f"{title} {body}".lower()
        
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "also", "now", "i", "we", "you",
            "they", "it", "this", "that", "these", "those", "what", "which", "who",
            "whom", "and", "but", "or", "if", "because", "while", "although",
            "after", "when", "however", "please", "thanks", "thank", "hi", "hello",
        }
        
        # Extract words (3+ characters)
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', content))
        
        return words - stop_words
    
    def _calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles using word overlap."""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate overlap
        intersection = len(words1 & words2)
        smaller_set = min(len(words1), len(words2))
        
        return intersection / smaller_set if smaller_set > 0 else 0.0
    
    def get_deduplication_report(self, context: AgentContext) -> str:
        """
        Generate a human-readable report of potential duplicates.
        """
        duplicates = context.metadata.get("potential_duplicates", [])
        similar = context.metadata.get("similar_issues", [])
        
        if not duplicates and not similar:
            return "No similar or duplicate issues found."
        
        lines = ["## Similar Issue Analysis\n"]
        
        if duplicates:
            lines.append("### ⚠️ Potential Duplicates\n")
            for dup in duplicates:
                lines.append(
                    f"- **#{dup['issue_number']}**: {dup['title']} "
                    f"(Score: {dup['similarity_score']:.0%})"
                )
            lines.append("")
        
        other_similar = [s for s in similar if not s.get("is_potential_duplicate")]
        if other_similar:
            lines.append("### 🔗 Related Issues\n")
            for issue in other_similar[:5]:
                lines.append(
                    f"- **#{issue['issue_number']}**: {issue['title']} "
                    f"(Score: {issue['similarity_score']:.0%})"
                )
        
        return "\n".join(lines)
