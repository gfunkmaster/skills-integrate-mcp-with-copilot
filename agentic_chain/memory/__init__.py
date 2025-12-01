"""
Memory System for Context Awareness.

Provides persistence and retrieval of project analysis results,
issue solutions, and context across CLI runs.

Key features:
- SQLite-based persistent storage
- Vector embeddings for semantic search
- Short-term, long-term, and entity memory types
- Memory summarization and pruning
- Export/import functionality
"""

from .types import (
    MemoryType,
    MemoryEntry,
    MemoryQuery,
    MemorySearchResult,
)

from .storage import (
    MemoryStorage,
    SQLiteStorage,
)

from .embeddings import (
    EmbeddingProvider,
    SimpleEmbedding,
)

from .manager import (
    MemoryManager,
)


__all__ = [
    # Types
    "MemoryType",
    "MemoryEntry",
    "MemoryQuery",
    "MemorySearchResult",
    # Storage
    "MemoryStorage",
    "SQLiteStorage",
    # Embeddings
    "EmbeddingProvider",
    "SimpleEmbedding",
    # Manager
    "MemoryManager",
]
