"""
Base Agent class for the agentic chain framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..interactive.handler import InteractionHandler


@dataclass
class LLMContext:
    """LLM-related context for tracking usage and responses."""
    
    provider: Optional[str] = None
    model: Optional[str] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    responses: list = field(default_factory=list)
    
    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float = 0.0,
    ):
        """Add usage from a request."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.estimated_cost += cost
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "response_count": len(self.responses),
        }


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
    llm_context: LLMContext = field(default_factory=LLMContext)
    interaction_handler: Optional["InteractionHandler"] = field(default=None, repr=False)
    
    @property
    def interactive_mode(self) -> bool:
        """Check if interactive mode is enabled."""
        return (
            self.interaction_handler is not None 
            and self.interaction_handler.enabled
        )
    
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
            "llm_usage": self.llm_context.to_dict(),
            "interactive_mode": self.interactive_mode,
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
