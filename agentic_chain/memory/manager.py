"""
Memory Manager - High-level interface for the memory system.

Provides a simple API for storing, retrieving, and searching memories
with automatic embedding generation and context awareness.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import (
    MemoryEntry,
    MemoryQuery,
    MemorySearchResult,
    MemoryStats,
    MemoryType,
    utc_now,
)
from .storage import MemoryStorage, SQLiteStorage
from .embeddings import EmbeddingProvider, SimpleEmbedding, get_embedding_provider


logger = logging.getLogger(__name__)


class MemoryManager:
    """
    High-level memory management interface.
    
    Provides context-aware memory storage and retrieval for the
    agentic chain framework. Supports:
    
    - Storing project analysis results
    - Remembering issue solutions
    - Learning from successful implementations
    - Semantic search across memories
    - Automatic summarization and pruning
    
    Example usage:
        # Create manager
        manager = MemoryManager()
        
        # Store a project analysis
        manager.store_project_analysis(
            project_path="/path/to/project",
            analysis={"languages": ["Python"], ...}
        )
        
        # Find relevant memories for a new issue
        memories = manager.get_relevant_context(
            query="login authentication bug",
            project_path="/path/to/project"
        )
        
        # Clean up old memories
        manager.prune(max_age_days=90)
    """
    
    # Default settings
    DEFAULT_SHORT_TERM_MAX_AGE_DAYS = 7
    DEFAULT_LONG_TERM_MAX_AGE_DAYS = 365
    DEFAULT_MAX_ENTRIES = 10000
    DEFAULT_MIN_IMPORTANCE = 0.1
    
    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        use_semantic_search: bool = True,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the memory manager.
        
        Args:
            storage: Storage backend. Defaults to SQLite.
            embedding_provider: Embedding provider. Defaults to simple embeddings.
            use_semantic_search: Whether to use semantic search.
            db_path: Path to the database file for SQLite storage.
        """
        self._storage = storage or SQLiteStorage(db_path=db_path)
        self._embedding_provider = embedding_provider or SimpleEmbedding()
        self._use_semantic_search = use_semantic_search
    
    @property
    def storage(self) -> MemoryStorage:
        """Get the storage backend."""
        return self._storage
    
    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        return self._embedding_provider
    
    # ========== Core Memory Operations ==========
    
    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        metadata: Optional[Dict[str, Any]] = None,
        project_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        generate_embedding: bool = True,
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: The content to remember
            memory_type: Type of memory
            metadata: Additional metadata
            project_path: Associated project path
            tags: Tags for categorization
            importance: Importance score (0-1)
            generate_embedding: Whether to generate an embedding
            
        Returns:
            The ID of the stored memory
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            project_path=project_path,
            tags=tags or [],
            importance=importance,
        )
        
        if generate_embedding and self._use_semantic_search:
            entry.embedding = self._embedding_provider.embed(content)
        
        return self._storage.store(entry)
    
    def recall(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            The memory entry if found
        """
        return self._storage.retrieve(memory_id)
    
    def search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        project_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemorySearchResult]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query text
            memory_types: Types of memory to search
            project_path: Filter by project
            tags: Filter by tags
            limit: Maximum results
            min_importance: Minimum importance score
            
        Returns:
            List of matching results with scores
        """
        memory_query = MemoryQuery(
            query_text=query,
            memory_types=memory_types,
            project_path=project_path,
            tags=tags,
            limit=limit,
            min_importance=min_importance,
            use_semantic=self._use_semantic_search,
        )
        
        results = self._storage.search(memory_query)
        
        # Enhance with semantic scoring if we have embeddings
        if self._use_semantic_search and query:
            query_embedding = self._embedding_provider.embed(query)
            results = self._enhance_with_semantic_scores(results, query_embedding)
        
        return results[:limit]
    
    def _enhance_with_semantic_scores(
        self,
        results: List[MemorySearchResult],
        query_embedding: List[float],
    ) -> List[MemorySearchResult]:
        """Enhance results with semantic similarity scores."""
        for result in results:
            if result.entry.embedding:
                semantic_score = self._embedding_provider.similarity(
                    query_embedding,
                    result.entry.embedding,
                )
                # Combine with existing score
                result.score = result.score * 0.3 + semantic_score * 0.7
                result.match_type = "semantic"
        
        # Re-sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def forget(self, memory_id: str) -> bool:
        """
        Delete a specific memory.
        
        Args:
            memory_id: The memory ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self._storage.delete(memory_id)
    
    def update(self, entry: MemoryEntry) -> bool:
        """
        Update an existing memory.
        
        Args:
            entry: The updated memory entry
            
        Returns:
            True if updated, False if not found
        """
        # Regenerate embedding if content changed
        if self._use_semantic_search:
            entry.embedding = self._embedding_provider.embed(entry.content)
        
        return self._storage.update(entry)
    
    # ========== Specialized Memory Operations ==========
    
    def store_project_analysis(
        self,
        project_path: str,
        analysis: Dict[str, Any],
        importance: float = 0.7,
    ) -> str:
        """
        Store project analysis results.
        
        Args:
            project_path: Path to the project
            analysis: Analysis results dict
            importance: Importance score
            
        Returns:
            Memory ID
        """
        # Create a summary for the embedding
        summary_parts = []
        
        if analysis.get("languages"):
            langs = list(analysis["languages"].keys())[:5]
            summary_parts.append(f"Languages: {', '.join(langs)}")
        
        if analysis.get("patterns", {}).get("framework"):
            summary_parts.append(f"Framework: {analysis['patterns']['framework']}")
        
        if analysis.get("structure", {}).get("file_count"):
            summary_parts.append(f"Files: {analysis['structure']['file_count']}")
        
        summary = "; ".join(summary_parts) if summary_parts else "Project analysis"
        
        return self.remember(
            content=json.dumps(analysis, indent=2),
            memory_type=MemoryType.PROJECT,
            metadata={"summary": summary},
            project_path=project_path,
            tags=["project_analysis"],
            importance=importance,
        )
    
    def store_issue_solution(
        self,
        issue_title: str,
        issue_body: str,
        solution: Dict[str, Any],
        project_path: Optional[str] = None,
        importance: float = 0.8,
    ) -> str:
        """
        Store an issue solution for future reference.
        
        Args:
            issue_title: Title of the issue
            issue_body: Body of the issue
            solution: Solution data
            project_path: Associated project
            importance: Importance score
            
        Returns:
            Memory ID
        """
        content = f"Issue: {issue_title}\n\n{issue_body}\n\nSolution:\n{json.dumps(solution, indent=2)}"
        
        # Extract tags from issue
        tags = ["issue_solution"]
        if solution.get("issue_type"):
            tags.append(solution["issue_type"])
        
        return self.remember(
            content=content,
            memory_type=MemoryType.ISSUE,
            metadata={
                "issue_title": issue_title,
                "solution_type": solution.get("issue_type"),
            },
            project_path=project_path,
            tags=tags,
            importance=importance,
        )
    
    def store_pattern(
        self,
        pattern_name: str,
        pattern_description: str,
        pattern_data: Dict[str, Any],
        project_path: Optional[str] = None,
    ) -> str:
        """
        Store a recurring pattern discovered in the codebase.
        
        Args:
            pattern_name: Name of the pattern
            pattern_description: Description
            pattern_data: Pattern details
            project_path: Associated project
            
        Returns:
            Memory ID
        """
        content = f"Pattern: {pattern_name}\n\n{pattern_description}\n\n{json.dumps(pattern_data, indent=2)}"
        
        return self.remember(
            content=content,
            memory_type=MemoryType.PATTERN,
            metadata={"pattern_name": pattern_name},
            project_path=project_path,
            tags=["pattern", pattern_name.lower().replace(" ", "_")],
            importance=0.6,
        )
    
    def store_entity(
        self,
        entity_type: str,
        entity_name: str,
        entity_data: Dict[str, Any],
        project_path: Optional[str] = None,
    ) -> str:
        """
        Store information about a specific entity (file, function, class).
        
        Args:
            entity_type: Type of entity (file, function, class)
            entity_name: Name of the entity
            entity_data: Entity details
            project_path: Associated project
            
        Returns:
            Memory ID
        """
        content = f"{entity_type.title()}: {entity_name}\n\n{json.dumps(entity_data, indent=2)}"
        
        return self.remember(
            content=content,
            memory_type=MemoryType.ENTITY,
            metadata={
                "entity_type": entity_type,
                "entity_name": entity_name,
            },
            project_path=project_path,
            tags=["entity", entity_type],
            importance=0.5,
        )
    
    # ========== Context Retrieval ==========
    
    def get_relevant_context(
        self,
        query: str,
        project_path: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """
        Get relevant context for a query.
        
        This is the main method for agents to retrieve memories
        relevant to their current task.
        
        Args:
            query: The query or task description
            project_path: Optional project to focus on
            limit: Maximum memories to return
            
        Returns:
            List of relevant memory entries
        """
        results = self.search(
            query=query,
            project_path=project_path,
            limit=limit * 2,  # Fetch more for filtering
        )
        
        # Return just the entries
        entries = [r.entry for r in results[:limit]]
        
        return entries
    
    def get_project_context(
        self,
        project_path: str,
    ) -> Optional[MemoryEntry]:
        """
        Get stored project analysis for a project.
        
        Args:
            project_path: Path to the project
            
        Returns:
            The project analysis memory if found
        """
        results = self.search(
            query="project analysis",
            memory_types=[MemoryType.PROJECT],
            project_path=project_path,
            limit=1,
        )
        
        if results:
            return results[0].entry
        return None
    
    def get_similar_issues(
        self,
        issue_title: str,
        issue_body: str,
        project_path: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """
        Find similar issues that have been solved before.
        
        Args:
            issue_title: Title of the current issue
            issue_body: Body of the current issue
            project_path: Optional project filter
            limit: Maximum results
            
        Returns:
            List of similar issue memories
        """
        query = f"{issue_title} {issue_body}"
        
        results = self.search(
            query=query,
            memory_types=[MemoryType.ISSUE],
            project_path=project_path,
            limit=limit,
            min_importance=0.3,
        )
        
        return [r.entry for r in results]
    
    # ========== Maintenance Operations ==========
    
    def prune(
        self,
        max_age_days: Optional[int] = None,
        max_entries: Optional[int] = None,
        min_importance: Optional[float] = None,
    ) -> int:
        """
        Clean up old or low-importance memories.
        
        Args:
            max_age_days: Remove entries older than this
            max_entries: Keep only this many entries
            min_importance: Remove entries below this importance
            
        Returns:
            Number of entries removed
        """
        return self._storage.prune(
            max_age_days=max_age_days,
            max_entries=max_entries,
            min_importance=min_importance,
        )
    
    def auto_prune(self):
        """
        Perform automatic pruning with default settings.
        
        Removes old short-term memories and low-importance entries.
        """
        # Prune old short-term memories
        self.prune(max_age_days=self.DEFAULT_SHORT_TERM_MAX_AGE_DAYS)
        
        # Keep within entry limit
        self.prune(max_entries=self.DEFAULT_MAX_ENTRIES)
        
        # Remove very low importance entries
        self.prune(min_importance=self.DEFAULT_MIN_IMPORTANCE)
    
    def summarize_old_entries(
        self,
        max_age_days: int = 30,
        summarizer: Optional[callable] = None,
    ) -> int:
        """
        Summarize old entries to save space.
        
        Args:
            max_age_days: Age threshold for summarization
            summarizer: Optional custom summarization function
            
        Returns:
            Number of entries summarized
        """
        cutoff = utc_now() - timedelta(days=max_age_days)
        
        query = MemoryQuery(
            since=cutoff,
            limit=1000,
        )
        
        results = self._storage.search(query)
        summarized = 0
        
        for result in results:
            entry = result.entry
            if entry.summary:
                continue  # Already summarized
            
            # Generate summary
            if summarizer:
                entry.summary = summarizer(entry.content)
            else:
                entry.summary = self._simple_summarize(entry.content)
            
            self._storage.update(entry)
            summarized += 1
        
        if summarized > 0:
            logger.info(f"Summarized {summarized} old memory entries")
        
        return summarized
    
    def _simple_summarize(self, content: str, max_length: int = 200) -> str:
        """Simple summarization by truncation."""
        if len(content) <= max_length:
            return content
        
        # Find a good break point
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        break_point = max(last_period, last_newline)
        if break_point > max_length // 2:
            return content[:break_point + 1]
        
        return truncated + "..."
    
    # ========== Import/Export ==========
    
    def export_memories(self, output_path: str) -> int:
        """
        Export all memories to a JSON file.
        
        Args:
            output_path: Path to the output file
            
        Returns:
            Number of entries exported
        """
        entries = self._storage.export_all()
        
        with open(output_path, 'w') as f:
            json.dump({
                "version": 1,
                "exported_at": utc_now().isoformat(),
                "entries": entries,
            }, f, indent=2)
        
        logger.info(f"Exported {len(entries)} memories to {output_path}")
        return len(entries)
    
    def import_memories(self, input_path: str) -> int:
        """
        Import memories from a JSON file.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Number of entries imported
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        entries = data.get("entries", [])
        imported = self._storage.import_entries(entries)
        
        logger.info(f"Imported {imported} memories from {input_path}")
        return imported
    
    # ========== Statistics ==========
    
    def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        return self._storage.get_stats()
    
    def get_stats_summary(self) -> str:
        """Get a human-readable stats summary."""
        stats = self.get_stats()
        
        lines = [
            "📊 Memory System Statistics",
            "=" * 40,
            f"Total entries: {stats.total_entries}",
            f"Storage size: {stats.total_size_bytes / 1024:.1f} KB",
            "",
            "Entries by type:",
        ]
        
        for mem_type, count in stats.entries_by_type.items():
            lines.append(f"  • {mem_type}: {count}")
        
        lines.extend([
            "",
            f"Average importance: {stats.average_importance:.2f}",
            f"Total accesses: {stats.total_access_count}",
        ])
        
        if stats.oldest_entry:
            if isinstance(stats.oldest_entry, str):
                lines.append(f"Oldest entry: {stats.oldest_entry[:10]}")
            else:
                lines.append(f"Oldest entry: {stats.oldest_entry.date()}")
        if stats.newest_entry:
            if isinstance(stats.newest_entry, str):
                lines.append(f"Newest entry: {stats.newest_entry[:10]}")
            else:
                lines.append(f"Newest entry: {stats.newest_entry.date()}")
        
        return "\n".join(lines)
    
    def clear_all(self):
        """Clear all memories. Use with caution!"""
        if hasattr(self._storage, 'clear_all'):
            self._storage.clear_all()
    
    def close(self):
        """Close the memory manager and release resources."""
        if hasattr(self._storage, 'close'):
            self._storage.close()
