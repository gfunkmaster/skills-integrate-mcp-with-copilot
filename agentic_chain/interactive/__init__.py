"""
Interactive Mode for Human-in-the-Loop Processing.

Provides interactive mode capabilities where agents can request
human feedback at key decision points.

Key features:
- Interactive prompts at decision points
- Solution review before implementation
- Clarification requests when requirements are unclear
- Risk assessment for critical operations
- Alternative selection for multiple solutions
- Interaction history tracking
"""

from .types import (
    InteractionType,
    InteractionPoint,
    InteractionOption,
    InteractionResult,
    InteractionHistory,
)

from .handler import (
    InteractionHandler,
    ConsoleInteractionHandler,
)


__all__ = [
    # Types
    "InteractionType",
    "InteractionPoint",
    "InteractionOption",
    "InteractionResult",
    "InteractionHistory",
    # Handlers
    "InteractionHandler",
    "ConsoleInteractionHandler",
]
