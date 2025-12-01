"""
Type definitions for the interactive mode system.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> Optional[datetime]:
    """
    Parse a datetime from string or return as-is if already datetime.
    
    Handles ISO format strings including 'Z' suffix for UTC.
    
    Args:
        value: String or datetime to parse
        
    Returns:
        Parsed datetime or None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle 'Z' suffix for UTC
        if value.endswith('Z'):
            value = value[:-1] + '+00:00'
        return datetime.fromisoformat(value)
    return None


class InteractionType(Enum):
    """Types of interaction points."""
    
    # Review proposed solution before implementation
    SOLUTION_REVIEW = "solution_review"
    
    # Request clarification on unclear requirements
    CLARIFICATION = "clarification"
    
    # Request human judgment on risky operations
    RISK_ASSESSMENT = "risk_assessment"
    
    # Present multiple solutions for user choice
    ALTERNATIVE_SELECTION = "alternative_selection"
    
    # Confirm before proceeding with an action
    CONFIRMATION = "confirmation"
    
    # Request additional input or context
    INPUT_REQUEST = "input_request"


@dataclass
class InteractionOption:
    """
    An option presented to the user during interaction.
    
    Attributes:
        id: Unique identifier for the option
        label: Short label for the option
        description: Detailed description
        risk_level: Risk level (low, medium, high)
        estimated_time: Estimated time for this option
        metadata: Additional option metadata
    """
    
    id: str
    label: str
    description: str = ""
    risk_level: str = "low"
    estimated_time: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "risk_level": self.risk_level,
            "estimated_time": self.estimated_time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionOption":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            description=data.get("description", ""),
            risk_level=data.get("risk_level", "low"),
            estimated_time=data.get("estimated_time"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InteractionPoint:
    """
    A point where human interaction is requested.
    
    Attributes:
        interaction_type: Type of interaction
        title: Title/heading for the interaction
        message: Main message to display
        options: Available options (for selection types)
        context: Additional context information
        default_option: Default option ID if user provides no input
        allow_custom_input: Whether to allow custom text input
        timeout_seconds: Optional timeout for the interaction
    """
    
    interaction_type: InteractionType
    title: str
    message: str
    options: List[InteractionOption] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    default_option: Optional[str] = None
    allow_custom_input: bool = True
    timeout_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interaction_type": self.interaction_type.value,
            "title": self.title,
            "message": self.message,
            "options": [opt.to_dict() for opt in self.options],
            "context": self.context,
            "default_option": self.default_option,
            "allow_custom_input": self.allow_custom_input,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionPoint":
        """Create from dictionary."""
        interaction_type_str = data.get("interaction_type", "confirmation")
        interaction_type = InteractionType(interaction_type_str)
        
        options = [
            InteractionOption.from_dict(opt) 
            for opt in data.get("options", [])
        ]
        
        return cls(
            interaction_type=interaction_type,
            title=data.get("title", ""),
            message=data.get("message", ""),
            options=options,
            context=data.get("context", {}),
            default_option=data.get("default_option"),
            allow_custom_input=data.get("allow_custom_input", True),
            timeout_seconds=data.get("timeout_seconds"),
        )


@dataclass
class InteractionResult:
    """
    Result of a user interaction.
    
    Attributes:
        approved: Whether the user approved/accepted
        selected_option: ID of selected option (if applicable)
        custom_input: Custom text input from user
        feedback: Additional feedback provided
        timestamp: When the interaction occurred
        skipped: Whether interaction was skipped (e.g., timeout)
    """
    
    approved: bool = True
    selected_option: Optional[str] = None
    custom_input: Optional[str] = None
    feedback: Optional[str] = None
    timestamp: Optional[datetime] = None
    skipped: bool = False
    
    def __post_init__(self):
        """Initialize defaults after creation."""
        if self.timestamp is None:
            self.timestamp = utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "selected_option": self.selected_option,
            "custom_input": self.custom_input,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "skipped": self.skipped,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionResult":
        """Create from dictionary."""
        return cls(
            approved=data.get("approved", True),
            selected_option=data.get("selected_option"),
            custom_input=data.get("custom_input"),
            feedback=data.get("feedback"),
            timestamp=parse_datetime(data.get("timestamp")),
            skipped=data.get("skipped", False),
        )


@dataclass
class InteractionRecord:
    """
    Record of a single interaction for history tracking.
    
    Attributes:
        id: Unique identifier
        point: The interaction point
        result: The result of the interaction
        agent_name: Name of the agent that requested interaction
        issue_context: Context about the current issue
    """
    
    id: Optional[str] = None
    point: Optional[InteractionPoint] = None
    result: Optional[InteractionResult] = None
    agent_name: Optional[str] = None
    issue_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            import hashlib
            timestamp = utc_now().isoformat()
            content = f"{self.agent_name}{timestamp}"
            self.id = f"int_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "point": self.point.to_dict() if self.point else None,
            "result": self.result.to_dict() if self.result else None,
            "agent_name": self.agent_name,
            "issue_context": self.issue_context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRecord":
        """Create from dictionary."""
        point = None
        if data.get("point"):
            point = InteractionPoint.from_dict(data["point"])
        
        result = None
        if data.get("result"):
            result = InteractionResult.from_dict(data["result"])
        
        return cls(
            id=data.get("id"),
            point=point,
            result=result,
            agent_name=data.get("agent_name"),
            issue_context=data.get("issue_context", {}),
        )


@dataclass
class InteractionHistory:
    """
    History of interactions during a session.
    
    Attributes:
        session_id: Unique session identifier
        records: List of interaction records
        started_at: When the session started
        completed_at: When the session completed
        metadata: Additional session metadata
    """
    
    session_id: Optional[str] = None
    records: List[InteractionRecord] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.started_at is None:
            self.started_at = utc_now()
        if self.session_id is None:
            import hashlib
            content = f"session_{self.started_at.isoformat()}"
            self.session_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_record(self, record: InteractionRecord):
        """Add an interaction record."""
        self.records.append(record)
    
    def complete(self):
        """Mark the session as completed."""
        self.completed_at = utc_now()
    
    @property
    def record_count(self) -> int:
        """Get the number of records."""
        return len(self.records)
    
    @property
    def approved_count(self) -> int:
        """Get count of approved interactions."""
        return sum(1 for r in self.records if r.result and r.result.approved)
    
    @property
    def rejected_count(self) -> int:
        """Get count of rejected interactions."""
        return sum(1 for r in self.records if r.result and not r.result.approved)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "records": [r.to_dict() for r in self.records],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
            "record_count": self.record_count,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionHistory":
        """Create from dictionary."""
        records = [
            InteractionRecord.from_dict(r) 
            for r in data.get("records", [])
        ]
        
        return cls(
            session_id=data.get("session_id"),
            records=records,
            started_at=parse_datetime(data.get("started_at")),
            completed_at=parse_datetime(data.get("completed_at")),
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "InteractionHistory":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
