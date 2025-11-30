"""
Base LLM Provider - Abstract base class and data structures for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional
from enum import Enum


class MessageRole(str, Enum):
    """Role of the message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """Represents a message in an LLM conversation."""
    role: MessageRole
    content: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class LLMUsage:
    """Token usage information for an LLM request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost estimation (in USD)
    estimated_cost: float = 0.0
    
    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Add usage from multiple requests."""
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: LLMUsage
    finish_reason: Optional[str] = None
    raw_response: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "estimated_cost": self.usage.estimated_cost,
            },
            "finish_reason": self.finish_reason,
        }


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: str  # "openai", "anthropic", "local"
    model: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Additional provider-specific options
    extra_options: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default models based on provider if not specified."""
        if not self.model:
            default_models = {
                "openai": "gpt-4",
                "anthropic": "claude-3-sonnet-20240229",
                "local": "llama2",
            }
            self.model = default_models.get(self.provider, "gpt-4")


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass


class LLMContextLengthError(LLMError):
    """Raised when context length is exceeded."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration for the provider.
        """
        self.config = config
        self._total_usage = LLMUsage()
    
    @property
    def total_usage(self) -> LLMUsage:
        """Get total usage across all requests."""
        return self._total_usage
    
    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional provider-specific options.
            
        Returns:
            LLMResponse with the generated content.
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream a completion for the given messages.
        
        Args:
            messages: List of messages in the conversation.
            **kwargs: Additional provider-specific options.
            
        Yields:
            Content chunks as they are generated.
        """
        pass
    
    def generate_code(
        self,
        prompt: str,
        context: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate code based on a prompt.
        
        Args:
            prompt: Description of the code to generate.
            context: Optional code context or file contents.
            language: Optional programming language.
            **kwargs: Additional options.
            
        Returns:
            LLMResponse with the generated code.
        """
        system_message = self._build_code_generation_system_prompt(language)
        user_message = self._build_code_generation_user_prompt(prompt, context)
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=system_message),
            LLMMessage(role=MessageRole.USER, content=user_message),
        ]
        
        return self.complete(messages, **kwargs)
    
    def review_code(
        self,
        code: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Review code and provide suggestions.
        
        Args:
            code: Code to review.
            context: Optional context about the code.
            **kwargs: Additional options.
            
        Returns:
            LLMResponse with review suggestions.
        """
        system_message = """You are an expert code reviewer. Analyze the provided code and provide:
1. Summary of what the code does
2. Potential bugs or issues
3. Security concerns
4. Performance improvements
5. Code quality suggestions

Format your response as structured feedback."""

        user_content = f"Please review this code:\n\n```\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n{context}"
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=system_message),
            LLMMessage(role=MessageRole.USER, content=user_content),
        ]
        
        return self.complete(messages, **kwargs)
    
    def generate_implementation_plan(
        self,
        issue_description: str,
        project_context: Optional[dict] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate an implementation plan for an issue.
        
        Args:
            issue_description: Description of the issue.
            project_context: Optional project analysis context.
            **kwargs: Additional options.
            
        Returns:
            LLMResponse with the implementation plan.
        """
        system_message = """You are an expert software architect. Create a detailed implementation plan for the given issue.
Include:
1. Problem analysis
2. Proposed solution approach
3. Files that need to be modified
4. Step-by-step implementation guide
5. Test cases to add
6. Potential risks and mitigations

Format your response with clear sections and actionable steps."""

        user_content = f"Issue to solve:\n{issue_description}"
        if project_context:
            user_content += f"\n\nProject context:\n"
            if project_context.get("languages"):
                user_content += f"Languages: {', '.join(project_context['languages'].keys())}\n"
            if project_context.get("patterns", {}).get("framework"):
                user_content += f"Framework: {project_context['patterns']['framework']}\n"
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=system_message),
            LLMMessage(role=MessageRole.USER, content=user_content),
        ]
        
        return self.complete(messages, **kwargs)
    
    def _build_code_generation_system_prompt(
        self,
        language: Optional[str] = None,
    ) -> str:
        """Build system prompt for code generation."""
        lang_hint = f" in {language}" if language else ""
        return f"""You are an expert programmer. Generate clean, well-documented, and efficient code{lang_hint}.
Follow best practices and include:
1. Clear comments explaining complex logic
2. Proper error handling
3. Type hints where applicable
4. Unit test suggestions

Output only the code with no additional explanation unless asked."""
    
    def _build_code_generation_user_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> str:
        """Build user prompt for code generation."""
        user_content = prompt
        if context:
            user_content = f"Context:\n```\n{context}\n```\n\nTask: {prompt}"
        return user_content
    
    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """
        Calculate estimated cost for the request.
        
        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            model: Model name.
            
        Returns:
            Estimated cost in USD.
        """
        # Pricing per 1000 tokens (approximate)
        pricing = {
            # OpenAI
            "gpt-4": (0.03, 0.06),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            # Anthropic
            "claude-3-opus": (0.015, 0.075),
            "claude-3-sonnet": (0.003, 0.015),
            "claude-3-haiku": (0.00025, 0.00125),
        }
        
        # Find matching pricing
        for model_prefix, (input_price, output_price) in pricing.items():
            if model_prefix in model.lower():
                input_cost = (prompt_tokens / 1000) * input_price
                output_cost = (completion_tokens / 1000) * output_price
                return input_cost + output_cost
        
        # Default pricing if model not found
        return 0.0
    
    def reset_usage(self):
        """Reset usage tracking."""
        self._total_usage = LLMUsage()
