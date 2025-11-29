"""
Orchestrator - Chains agents together to solve issues.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .agents import AgentContext, BaseAgent
from .agents.project_analyzer import ProjectAnalyzer
from .agents.issue_analyzer import IssueAnalyzer
from .agents.code_reviewer import CodeReviewer
from .agents.solution_implementer import SolutionImplementer


logger = logging.getLogger(__name__)


class AgenticChain:
    """
    Orchestrates a chain of agents to analyze projects, understand issues,
    review code, and propose solutions.
    
    Example usage:
        chain = AgenticChain(project_path="/path/to/project")
        result = chain.solve_issue({
            "title": "Bug in login",
            "body": "Users cannot login when..."
        })
    """
    
    def __init__(
        self,
        project_path: str,
        custom_agents: Optional[list] = None,
    ):
        """
        Initialize the agentic chain.
        
        Args:
            project_path: Path to the project to analyze.
            custom_agents: Optional list of custom agents to include.
        """
        self.project_path = Path(project_path).resolve()
        self.context = AgentContext(project_path=str(self.project_path))
        
        # Default agent pipeline
        self.agents = [
            ProjectAnalyzer(),
            IssueAnalyzer(),
            CodeReviewer(),
            SolutionImplementer(),
        ]
        
        # Add custom agents if provided
        if custom_agents:
            for agent in custom_agents:
                if isinstance(agent, BaseAgent):
                    self.agents.append(agent)
                    
        self._executed = False
        
    def add_agent(self, agent: BaseAgent, position: Optional[int] = None):
        """
        Add an agent to the chain.
        
        Args:
            agent: The agent to add.
            position: Optional position in the chain. If None, appends to end.
        """
        if position is not None:
            self.agents.insert(position, agent)
        else:
            self.agents.append(agent)
            
    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the chain by name.
        
        Args:
            agent_name: Name of the agent to remove.
            
        Returns:
            True if agent was removed, False if not found.
        """
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                self.agents.pop(i)
                return True
        return False
        
    def solve_issue(self, issue_data: dict) -> dict:
        """
        Run the full agentic chain to solve an issue.
        
        Args:
            issue_data: Dictionary containing issue information:
                - title: Issue title
                - body: Issue description
                - labels: Optional list of labels
                - number: Optional issue number
                
        Returns:
            Complete solution context including analysis and proposals.
        """
        self.context.issue_data = issue_data
        
        logger.info(f"Starting agentic chain for issue: {issue_data.get('title', 'Unknown')}")
        
        for agent in self.agents:
            logger.info(f"Executing agent: {agent.name}")
            try:
                self.context = agent.execute(self.context)
                logger.info(f"Agent {agent.name} completed successfully")
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {str(e)}")
                raise
                
        self._executed = True
        return self.get_result()
    
    def analyze_project(self) -> dict:
        """
        Run only project analysis without solving an issue.
        
        Returns:
            Project analysis results.
        """
        analyzer = ProjectAnalyzer()
        self.context = analyzer.execute(self.context)
        return self.context.project_analysis
    
    def get_result(self) -> dict:
        """
        Get the complete result of the agentic chain.
        
        Returns:
            Dictionary containing all analysis and solution data.
        """
        return self.context.to_dict()
    
    def get_solution_summary(self) -> str:
        """
        Get a human-readable summary of the solution.
        
        Returns:
            Formatted summary string.
        """
        if not self._executed:
            return "Chain has not been executed yet. Call solve_issue() first."
            
        result = self.get_result()
        
        lines = []
        lines.append("=" * 60)
        lines.append("AGENTIC CHAIN SOLUTION SUMMARY")
        lines.append("=" * 60)
        
        # Issue Analysis
        if result.get("issue_analysis"):
            analysis = result["issue_analysis"]
            lines.append("\n📋 ISSUE ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Title: {analysis.get('title', 'N/A')}")
            lines.append(f"Type: {analysis.get('issue_type', 'unknown')}")
            lines.append(f"Priority: {analysis.get('priority', 'medium')}")
            
            if analysis.get("requirements"):
                lines.append("\nRequirements:")
                for req in analysis["requirements"][:5]:
                    lines.append(f"  • {req[:80]}")
                    
        # Code Review
        if result.get("code_review"):
            review = result["code_review"]
            lines.append("\n🔍 CODE REVIEW")
            lines.append("-" * 40)
            
            if review.get("relevant_files"):
                lines.append(f"Relevant files: {len(review['relevant_files'])}")
                for f in review["relevant_files"][:5]:
                    lines.append(f"  • {f}")
                    
            if review.get("potential_issues"):
                lines.append(f"\nPotential issues found: {len(review['potential_issues'])}")
                
        # Solution
        if result.get("solution"):
            solution = result["solution"]
            lines.append("\n💡 PROPOSED SOLUTION")
            lines.append("-" * 40)
            
            if solution.get("implementation_plan"):
                plan = solution["implementation_plan"]
                lines.append(f"Complexity: {plan.get('complexity', 'N/A')}")
                lines.append(f"Estimated hours: {plan.get('estimated_hours', 'N/A')}")
                
            if solution.get("proposed_changes"):
                lines.append("\nProposed changes:")
                for change in solution["proposed_changes"][:3]:
                    lines.append(f"  • [{change.get('type', 'change')}] {change.get('description', '')[:60]}")
                    
            if solution.get("risks"):
                lines.append("\nRisks:")
                for risk in solution["risks"]:
                    lines.append(f"  • [{risk.get('level', 'N/A')}] {risk.get('description', '')[:60]}")
                    
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def export_result(self, output_path: str):
        """
        Export the result to a JSON file.
        
        Args:
            output_path: Path to save the JSON output.
        """
        result = self.get_result()
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
        logger.info(f"Result exported to {output_path}")
        
    def __repr__(self) -> str:
        agent_names = [a.name for a in self.agents]
        return f"AgenticChain(project='{self.project_path}', agents={agent_names})"
