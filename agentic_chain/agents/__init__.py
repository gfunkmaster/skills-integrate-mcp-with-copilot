"""
Base Agent class for the agentic chain framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentContext:
    """Shared context that agents can read from and write to."""
    
    project_path: str
    issue_data: Optional[dict] = None
    project_analysis: Optional[dict] = None
    issue_analysis: Optional[dict] = None
    code_review: Optional[dict] = None
    solution: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert context to dictionary."""
        return {
            "project_path": self.project_path,
            "issue_data": self.issue_data,
            "project_analysis": self.project_analysis,
            "issue_analysis": self.issue_analysis,
            "code_review": self.code_review,
            "solution": self.solution,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Base class for all agents in the chain."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute the agent's task and update the context.
        
        Args:
            context: The shared context object.
            
        Returns:
            Updated context with agent's results.
        """
        pass
    
    def validate_context(self, context: AgentContext) -> bool:
        """
        Validate that context has required data for this agent.
        
        Override in subclasses if specific validation is needed.
        """
        return context.project_path is not None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
