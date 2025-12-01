"""
Interaction handlers for human-in-the-loop processing.

Provides mechanisms for agents to pause and request human input
at key decision points during execution.
"""

import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .types import (
    InteractionType,
    InteractionPoint,
    InteractionOption,
    InteractionResult,
    InteractionRecord,
    InteractionHistory,
)


logger = logging.getLogger(__name__)


class InteractionHandler(ABC):
    """
    Abstract base class for interaction handlers.
    
    Subclasses implement specific interaction mechanisms
    (console, web UI, notifications, etc.)
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the handler.
        
        Args:
            enabled: Whether interactive mode is enabled
        """
        self.enabled = enabled
        self._history = InteractionHistory()
        self._callbacks: Dict[InteractionType, List[Callable]] = {}
    
    @property
    def history(self) -> InteractionHistory:
        """Get the interaction history."""
        return self._history
    
    def reset_history(self):
        """Reset the interaction history for a new session."""
        self._history = InteractionHistory()
    
    @abstractmethod
    def handle(
        self,
        point: InteractionPoint,
        agent_name: Optional[str] = None,
        issue_context: Optional[Dict[str, Any]] = None,
    ) -> InteractionResult:
        """
        Handle an interaction point.
        
        Args:
            point: The interaction point to handle
            agent_name: Name of the agent requesting interaction
            issue_context: Context about the current issue
            
        Returns:
            The result of the interaction
        """
        pass
    
    def request_confirmation(
        self,
        title: str,
        message: str,
        agent_name: Optional[str] = None,
        default: bool = True,
    ) -> InteractionResult:
        """
        Request a simple confirmation from the user.
        
        Args:
            title: Title for the confirmation
            message: Message to display
            agent_name: Name of the requesting agent
            default: Default value if no input
            
        Returns:
            InteractionResult with approved=True/False
        """
        point = InteractionPoint(
            interaction_type=InteractionType.CONFIRMATION,
            title=title,
            message=message,
            default_option="yes" if default else "no",
            allow_custom_input=False,
        )
        return self.handle(point, agent_name)
    
    def request_selection(
        self,
        title: str,
        message: str,
        options: List[InteractionOption],
        agent_name: Optional[str] = None,
        allow_custom: bool = True,
    ) -> InteractionResult:
        """
        Request the user to select from options.
        
        Args:
            title: Title for the selection
            message: Message to display
            options: Available options
            agent_name: Name of the requesting agent
            allow_custom: Whether to allow custom input
            
        Returns:
            InteractionResult with selected_option set
        """
        point = InteractionPoint(
            interaction_type=InteractionType.ALTERNATIVE_SELECTION,
            title=title,
            message=message,
            options=options,
            allow_custom_input=allow_custom,
        )
        return self.handle(point, agent_name)
    
    def request_input(
        self,
        title: str,
        message: str,
        agent_name: Optional[str] = None,
    ) -> InteractionResult:
        """
        Request custom input from the user.
        
        Args:
            title: Title for the input request
            message: Message/prompt to display
            agent_name: Name of the requesting agent
            
        Returns:
            InteractionResult with custom_input set
        """
        point = InteractionPoint(
            interaction_type=InteractionType.INPUT_REQUEST,
            title=title,
            message=message,
            allow_custom_input=True,
        )
        return self.handle(point, agent_name)
    
    def request_solution_review(
        self,
        title: str,
        solution_summary: str,
        proposed_changes: List[Dict[str, Any]],
        risks: List[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
    ) -> InteractionResult:
        """
        Request review of a proposed solution.
        
        Args:
            title: Title for the review
            solution_summary: Summary of the solution
            proposed_changes: List of proposed changes
            risks: List of identified risks
            agent_name: Name of the requesting agent
            
        Returns:
            InteractionResult with approval and feedback
        """
        context = {
            "proposed_changes": proposed_changes,
            "risks": risks or [],
        }
        
        point = InteractionPoint(
            interaction_type=InteractionType.SOLUTION_REVIEW,
            title=title,
            message=solution_summary,
            context=context,
            allow_custom_input=True,
        )
        return self.handle(point, agent_name)
    
    def request_risk_assessment(
        self,
        title: str,
        message: str,
        risk_level: str,
        risk_details: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> InteractionResult:
        """
        Request human judgment on a risky operation.
        
        Args:
            title: Title for the assessment
            message: Description of the risk
            risk_level: Risk level (low, medium, high)
            risk_details: Detailed risk information
            agent_name: Name of the requesting agent
            
        Returns:
            InteractionResult with approval
        """
        context = {
            "risk_level": risk_level,
            "risk_details": risk_details,
        }
        
        point = InteractionPoint(
            interaction_type=InteractionType.RISK_ASSESSMENT,
            title=title,
            message=message,
            context=context,
            allow_custom_input=True,
        )
        return self.handle(point, agent_name)
    
    def register_callback(
        self,
        interaction_type: InteractionType,
        callback: Callable[[InteractionPoint, InteractionResult], None],
    ):
        """
        Register a callback for a specific interaction type.
        
        Args:
            interaction_type: Type of interaction to listen for
            callback: Function to call with point and result
        """
        if interaction_type not in self._callbacks:
            self._callbacks[interaction_type] = []
        self._callbacks[interaction_type].append(callback)
    
    def _notify_callbacks(
        self,
        point: InteractionPoint,
        result: InteractionResult,
    ):
        """Notify registered callbacks."""
        callbacks = self._callbacks.get(point.interaction_type, [])
        for callback in callbacks:
            try:
                callback(point, result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def _record_interaction(
        self,
        point: InteractionPoint,
        result: InteractionResult,
        agent_name: Optional[str] = None,
        issue_context: Optional[Dict[str, Any]] = None,
    ):
        """Record an interaction in history."""
        record = InteractionRecord(
            point=point,
            result=result,
            agent_name=agent_name,
            issue_context=issue_context or {},
        )
        self._history.add_record(record)
        self._notify_callbacks(point, result)


class ConsoleInteractionHandler(InteractionHandler):
    """
    Console-based interaction handler.
    
    Provides interactive prompts via stdin/stdout for
    human-in-the-loop processing in CLI applications.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        input_func: Optional[Callable[[str], str]] = None,
        output_func: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the console handler.
        
        Args:
            enabled: Whether interactive mode is enabled
            input_func: Custom input function (defaults to input())
            output_func: Custom output function (defaults to print())
        """
        super().__init__(enabled)
        self._input = input_func or input
        self._output = output_func or print
    
    def handle(
        self,
        point: InteractionPoint,
        agent_name: Optional[str] = None,
        issue_context: Optional[Dict[str, Any]] = None,
    ) -> InteractionResult:
        """
        Handle an interaction point via console.
        
        If interactive mode is disabled, returns auto-approved result.
        """
        if not self.enabled:
            result = InteractionResult(
                approved=True,
                skipped=True,
                selected_option=point.default_option,
            )
            self._record_interaction(point, result, agent_name, issue_context)
            return result
        
        # Display the interaction
        self._display_interaction(point)
        
        # Get user input based on interaction type
        if point.interaction_type == InteractionType.CONFIRMATION:
            result = self._handle_confirmation(point)
        elif point.interaction_type == InteractionType.ALTERNATIVE_SELECTION:
            result = self._handle_selection(point)
        elif point.interaction_type == InteractionType.INPUT_REQUEST:
            result = self._handle_input_request(point)
        elif point.interaction_type == InteractionType.SOLUTION_REVIEW:
            result = self._handle_solution_review(point)
        elif point.interaction_type == InteractionType.RISK_ASSESSMENT:
            result = self._handle_risk_assessment(point)
        elif point.interaction_type == InteractionType.CLARIFICATION:
            result = self._handle_clarification(point)
        else:
            result = self._handle_default(point)
        
        self._record_interaction(point, result, agent_name, issue_context)
        return result
    
    def _display_interaction(self, point: InteractionPoint):
        """Display the interaction prompt."""
        self._output("")
        self._output("=" * 60)
        self._output(f"🤖 {point.title}")
        self._output("=" * 60)
        self._output("")
        self._output(point.message)
        
        if point.options:
            self._output("")
            self._output("Options:")
            for i, opt in enumerate(point.options, 1):
                risk_indicator = self._get_risk_indicator(opt.risk_level)
                time_str = f" ({opt.estimated_time})" if opt.estimated_time else ""
                self._output(f"  {i}. {opt.label} {risk_indicator}{time_str}")
                if opt.description:
                    self._output(f"     {opt.description}")
        
        if point.context:
            if point.context.get("risks"):
                self._output("")
                self._output("⚠️  Identified Risks:")
                for risk in point.context["risks"]:
                    level = risk.get("level", "unknown")
                    desc = risk.get("description", "")
                    self._output(f"  [{level.upper()}] {desc}")
        
        self._output("")
    
    def _get_risk_indicator(self, risk_level: str) -> str:
        """Get a risk indicator string."""
        indicators = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴",
        }
        return indicators.get(risk_level.lower(), "⚪")
    
    def _handle_confirmation(self, point: InteractionPoint) -> InteractionResult:
        """Handle a confirmation interaction."""
        default_str = "Y/n" if point.default_option == "yes" else "y/N"
        
        try:
            response = self._input(f"Proceed? [{default_str}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        if not response:
            approved = point.default_option == "yes"
        else:
            approved = response in ("y", "yes", "1", "true")
        
        return InteractionResult(approved=approved)
    
    def _handle_selection(self, point: InteractionPoint) -> InteractionResult:
        """Handle a selection interaction."""
        prompt = "Enter selection"
        if point.allow_custom_input:
            prompt += " (number or custom input)"
        prompt += ": "
        
        try:
            response = self._input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        if not response and point.default_option:
            return InteractionResult(
                approved=True,
                selected_option=point.default_option,
            )
        
        # Check if response is a number
        try:
            selection_idx = int(response) - 1
            if 0 <= selection_idx < len(point.options):
                selected = point.options[selection_idx]
                return InteractionResult(
                    approved=True,
                    selected_option=selected.id,
                )
        except ValueError:
            pass
        
        # Treat as custom input
        if point.allow_custom_input and response:
            return InteractionResult(
                approved=True,
                custom_input=response,
            )
        
        # Invalid selection
        self._output("Invalid selection. Please try again.")
        return self._handle_selection(point)
    
    def _handle_input_request(self, point: InteractionPoint) -> InteractionResult:
        """Handle an input request interaction."""
        try:
            response = self._input("Your input: ").strip()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        return InteractionResult(
            approved=bool(response),
            custom_input=response if response else None,
        )
    
    def _handle_solution_review(self, point: InteractionPoint) -> InteractionResult:
        """Handle a solution review interaction."""
        # Show proposed changes if available
        if point.context.get("proposed_changes"):
            self._output("Proposed Changes:")
            for change in point.context["proposed_changes"]:
                change_type = change.get("type", "change")
                desc = change.get("description", "")
                self._output(f"  • [{change_type}] {desc}")
            self._output("")
        
        try:
            response = self._input("Approve this solution? [Y/n/modify]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        if not response or response in ("y", "yes"):
            return InteractionResult(approved=True)
        elif response in ("n", "no"):
            return InteractionResult(approved=False)
        else:
            # User wants to modify - get feedback
            try:
                feedback = self._input("Enter your modifications or feedback: ").strip()
            except (EOFError, KeyboardInterrupt):
                return InteractionResult(approved=False, skipped=True)
            
            return InteractionResult(
                approved=True,
                custom_input=feedback,
                feedback=feedback,
            )
    
    def _handle_risk_assessment(self, point: InteractionPoint) -> InteractionResult:
        """Handle a risk assessment interaction."""
        risk_level = point.context.get("risk_level", "unknown")
        
        if risk_level.lower() == "high":
            prompt = "This is a HIGH RISK operation. Are you sure? [yes/NO]: "
            default_approved = False
        else:
            prompt = "Proceed with this operation? [Y/n]: "
            default_approved = True
        
        try:
            response = self._input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        if not response:
            approved = default_approved
        elif risk_level.lower() == "high":
            approved = response == "yes"  # Require full "yes" for high risk
        else:
            approved = response in ("y", "yes")
        
        return InteractionResult(approved=approved)
    
    def _handle_clarification(self, point: InteractionPoint) -> InteractionResult:
        """Handle a clarification request."""
        try:
            response = self._input("Please provide clarification: ").strip()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        return InteractionResult(
            approved=bool(response),
            custom_input=response if response else None,
            feedback=response if response else None,
        )
    
    def _handle_default(self, point: InteractionPoint) -> InteractionResult:
        """Handle an unknown interaction type."""
        try:
            response = self._input("Continue? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return InteractionResult(approved=False, skipped=True)
        
        approved = not response or response in ("y", "yes")
        return InteractionResult(approved=approved)


class AutoApproveHandler(InteractionHandler):
    """
    Handler that automatically approves all interactions.
    
    Used when running in non-interactive (automated) mode.
    """
    
    def __init__(self):
        """Initialize with disabled state."""
        super().__init__(enabled=False)
    
    def handle(
        self,
        point: InteractionPoint,
        agent_name: Optional[str] = None,
        issue_context: Optional[Dict[str, Any]] = None,
    ) -> InteractionResult:
        """Auto-approve all interactions."""
        result = InteractionResult(
            approved=True,
            skipped=True,
            selected_option=point.default_option or (
                point.options[0].id if point.options else None
            ),
        )
        self._record_interaction(point, result, agent_name, issue_context)
        return result
