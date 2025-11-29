"""
Issue Analyzer Agent - Parses and understands issues to extract actionable requirements.
"""

import re
from typing import Optional

from . import BaseAgent, AgentContext


class IssueAnalyzer(BaseAgent):
    """
    Analyzes issues to understand requirements, affected areas,
    and priority for resolution.
    """
    
    def __init__(self, name: str = "IssueAnalyzer"):
        super().__init__(name)
        
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Analyze the issue and extract actionable requirements.
        """
        if not context.issue_data:
            raise ValueError("Issue data is required for analysis")
            
        issue = context.issue_data
        
        analysis = {
            "title": issue.get("title", ""),
            "issue_type": self._classify_issue(issue),
            "priority": self._determine_priority(issue),
            "affected_files": self._extract_file_references(issue),
            "requirements": self._extract_requirements(issue),
            "acceptance_criteria": self._extract_acceptance_criteria(issue),
            "keywords": self._extract_keywords(issue),
            "related_components": self._identify_components(issue, context),
        }
        
        context.issue_analysis = analysis
        return context
    
    def _classify_issue(self, issue: dict) -> str:
        """Classify the issue type."""
        title = issue.get("title", "").lower()
        body = issue.get("body", "").lower()
        labels = [l.get("name", "").lower() for l in issue.get("labels", [])]
        
        # Check labels first
        if any("bug" in label for label in labels):
            return "bug"
        if any("feature" in label for label in labels):
            return "feature"
        if any("enhancement" in label for label in labels):
            return "enhancement"
        if any("doc" in label for label in labels):
            return "documentation"
        if any("security" in label for label in labels):
            return "security"
        if any("performance" in label for label in labels):
            return "performance"
            
        # Analyze content
        content = f"{title} {body}"
        
        bug_keywords = ["bug", "error", "fix", "broken", "crash", "fail", "issue", "problem"]
        feature_keywords = ["add", "new", "implement", "create", "feature", "request"]
        enhancement_keywords = ["improve", "enhance", "update", "refactor", "optimize"]
        doc_keywords = ["document", "readme", "guide", "tutorial", "docs"]
        
        if any(kw in content for kw in bug_keywords):
            return "bug"
        if any(kw in content for kw in feature_keywords):
            return "feature"
        if any(kw in content for kw in enhancement_keywords):
            return "enhancement"
        if any(kw in content for kw in doc_keywords):
            return "documentation"
            
        return "unknown"
    
    def _determine_priority(self, issue: dict) -> str:
        """Determine issue priority."""
        labels = [l.get("name", "").lower() for l in issue.get("labels", [])]
        
        priority_indicators = {
            "critical": ["critical", "urgent", "p0", "blocker", "security"],
            "high": ["high", "important", "p1"],
            "medium": ["medium", "p2", "normal"],
            "low": ["low", "p3", "minor"],
        }
        
        for priority, indicators in priority_indicators.items():
            if any(ind in " ".join(labels) for ind in indicators):
                return priority
                
        # Default based on issue type
        title = issue.get("title", "").lower()
        if "security" in title or "vulnerability" in title:
            return "critical"
        if "crash" in title or "data loss" in title:
            return "high"
            
        return "medium"
    
    def _extract_file_references(self, issue: dict) -> list:
        """Extract file path references from issue content."""
        body = issue.get("body", "")
        
        # Match common file path patterns
        patterns = [
            r'`([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z]{1,5})`',  # `path/to/file.ext`
            r'(?:in|at|file|path)\s+["\']?([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z]{1,5})["\']?',
            r'([a-zA-Z0-9_\-]+\.(?:py|js|ts|go|rs|java|rb|cpp|c|cs|html|css|json|yaml|yml|md))',
        ]
        
        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, body, re.IGNORECASE)
            files.update(matches)
            
        return list(files)
    
    def _extract_requirements(self, issue: dict) -> list:
        """Extract requirements from issue content."""
        body = issue.get("body", "")
        requirements = []
        
        # Look for bullet points or numbered lists
        list_patterns = [
            r'^[\*\-\+]\s+(.+)$',  # Bullet points
            r'^\d+\.\s+(.+)$',     # Numbered lists
            r'^-\s*\[\s*\]\s+(.+)$',  # Checkboxes
        ]
        
        for line in body.split('\n'):
            line = line.strip()
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    req = match.group(1).strip()
                    if len(req) > 5 and not req.startswith('#'):
                        requirements.append(req)
                        
        # If no list items found, try to extract from sentences
        if not requirements:
            action_words = ["should", "must", "need", "require", "want"]
            for sentence in re.split(r'[.!?]', body):
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in action_words):
                    if len(sentence) > 10:
                        requirements.append(sentence)
                        
        return requirements[:10]  # Limit to 10 requirements
    
    def _extract_acceptance_criteria(self, issue: dict) -> list:
        """Extract acceptance criteria from issue content."""
        body = issue.get("body", "")
        criteria = []
        
        # Look for acceptance criteria section
        ac_patterns = [
            r'(?:acceptance criteria|ac|done when|definition of done)[:\s]*\n(.+?)(?:\n\n|\Z)',
            r'(?:expected behavior|expected result|should)[:\s]*\n(.+?)(?:\n\n|\Z)',
        ]
        
        for pattern in ac_patterns:
            match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
            if match:
                ac_section = match.group(1)
                # Extract bullet points
                for line in ac_section.split('\n'):
                    line = line.strip()
                    if line.startswith(('-', '*', '+')) or re.match(r'^\d+\.', line):
                        criteria.append(re.sub(r'^[\*\-\+\d\.]+\s*', '', line))
                        
        return criteria
    
    def _extract_keywords(self, issue: dict) -> list:
        """Extract relevant keywords from issue."""
        title = issue.get("title", "")
        body = issue.get("body", "")
        content = f"{title} {body}".lower()
        
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "also", "now", "i", "we", "you", "they", "it", "this",
            "that", "these", "those", "what", "which", "who", "whom", "and", "but",
            "or", "if", "because", "while", "although", "after", "when", "however",
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:15]]
    
    def _identify_components(self, issue: dict, context: AgentContext) -> list:
        """Identify affected components based on project analysis."""
        components = []
        
        if context.project_analysis:
            project = context.project_analysis
            body = issue.get("body", "").lower()
            title = issue.get("title", "").lower()
            content = f"{title} {body}"
            
            # Check against detected languages
            if project.get("languages"):
                for lang in project["languages"]:
                    if lang.lower() in content:
                        components.append(f"language:{lang}")
                        
            # Check against framework
            if project.get("patterns", {}).get("framework"):
                framework = project["patterns"]["framework"]
                if framework.lower() in content:
                    components.append(f"framework:{framework}")
                    
            # Check against directories
            if project.get("structure", {}).get("directories"):
                for directory in project["structure"]["directories"]:
                    if directory.lower() in content:
                        components.append(f"directory:{directory}")
                        
        return components
