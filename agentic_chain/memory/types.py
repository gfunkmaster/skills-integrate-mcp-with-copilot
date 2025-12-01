"""
Memory type definitions for the memory system.
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


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


class MemoryType(Enum):
    """Types of memory entries."""
    
    # Short-term memory: Recent context, temporary data
    SHORT_TERM = "short_term"
    
    # Long-term memory: Persisted knowledge, project insights
    LONG_TERM = "long_term"
    
    # Entity memory: Information about specific entities (files, functions, issues)
    ENTITY = "entity"
    
    # Project memory: Project-specific analysis results
    PROJECT = "project"
    
    # Issue memory: Previous issue solutions
    ISSUE = "issue"
    
    # Pattern memory: Recurring patterns in codebase
    PATTERN = "pattern"


@dataclass
class MemoryEntry:
    """
    A single memory entry that can be stored and retrieved.
    
    Attributes:
        content: The main content of the memory
        memory_type: Type of memory (short-term, long-term, entity)
        metadata: Additional metadata about the memory
        embedding: Optional vector embedding for semantic search
        created_at: When the memory was created
        updated_at: When the memory was last updated
        access_count: Number of times this memory has been accessed
        importance: Importance score (0-1) for pruning decisions
        project_path: Associated project path
        tags: Tags for categorization
    """
    
    content: str
    memory_type: MemoryType
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    access_count: int = 0
    importance: float = 0.5
    project_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    
    def __post_init__(self):
        """Initialize defaults after creation."""
        if self.created_at is None:
            self.created_at = utc_now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this memory entry."""
        content_hash = hashlib.sha256(
            f"{self.content}{self.memory_type.value}{self.created_at}".encode()
        ).hexdigest()[:16]
        return f"mem_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "access_count": self.access_count,
            "importance": self.importance,
            "project_path": self.project_path,
            "tags": self.tags,
            "summary": self.summary,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        created_at = parse_datetime(data.get("created_at"))
        updated_at = parse_datetime(data.get("updated_at"))
        
        memory_type_str = data.get("memory_type", "long_term")
        memory_type = MemoryType(memory_type_str)
        
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            memory_type=memory_type,
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=created_at,
            updated_at=updated_at,
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            project_path=data.get("project_path"),
            tags=data.get("tags", []),
            summary=data.get("summary"),
        )
    
    def touch(self):
        """Update access time and count."""
        self.access_count += 1
        self.updated_at = utc_now()


@dataclass
class MemoryQuery:
    """
    A query for retrieving memories.
    
    Attributes:
        query_text: Text to search for
        memory_types: Types of memory to search
        project_path: Filter by project path
        tags: Filter by tags
        min_importance: Minimum importance score
        limit: Maximum number of results
        use_semantic: Whether to use semantic search
    """
    
    query_text: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    project_path: Optional[str] = None
    tags: Optional[List[str]] = None
    min_importance: float = 0.0
    limit: int = 10
    use_semantic: bool = True
    since: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_text": self.query_text,
            "memory_types": [mt.value for mt in self.memory_types] if self.memory_types else None,
            "project_path": self.project_path,
            "tags": self.tags,
            "min_importance": self.min_importance,
            "limit": self.limit,
            "use_semantic": self.use_semantic,
            "since": self.since.isoformat() if self.since else None,
        }


@dataclass
class MemorySearchResult:
    """
    Result of a memory search.
    
    Attributes:
        entry: The memory entry
        score: Relevance score (higher is better)
        match_type: How the match was found (semantic, keyword, etc.)
    """
    
    entry: MemoryEntry
    score: float = 0.0
    match_type: str = "keyword"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry": self.entry.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
        }


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    
    total_entries: int = 0
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None
    average_importance: float = 0.0
    total_access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entries": self.total_entries,
            "entries_by_type": self.entries_by_type,
            "total_size_bytes": self.total_size_bytes,
            "oldest_entry": self.oldest_entry.isoformat() if self.oldest_entry else None,
            "newest_entry": self.newest_entry.isoformat() if self.newest_entry else None,
            "average_importance": self.average_importance,
            "total_access_count": self.total_access_count,
        }
