"""
Memory storage backends.

Provides SQLite-based persistent storage for memory entries.
"""

import json
import logging
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .types import MemoryEntry, MemoryQuery, MemorySearchResult, MemoryStats, MemoryType, utc_now


logger = logging.getLogger(__name__)


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry.
        
        Args:
            entry: The memory entry to store
            
        Returns:
            The ID of the stored entry
        """
        pass
    
    @abstractmethod
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.
        
        Args:
            entry_id: The ID of the entry to retrieve
            
        Returns:
            The memory entry if found, None otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """
        Search for memory entries matching a query.
        
        Args:
            query: The search query
            
        Returns:
            List of matching results with scores
        """
        pass
    
    @abstractmethod
    def update(self, entry: MemoryEntry) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry: The entry to update
            
        Returns:
            True if the entry was updated, False if not found
        """
        pass
    
    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The ID of the entry to delete
            
        Returns:
            True if the entry was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        pass
    
    @abstractmethod
    def prune(
        self,
        max_age_days: Optional[int] = None,
        max_entries: Optional[int] = None,
        min_importance: Optional[float] = None,
    ) -> int:
        """
        Prune old or low-importance entries.
        
        Args:
            max_age_days: Remove entries older than this
            max_entries: Keep only this many entries (by importance)
            min_importance: Remove entries below this importance
            
        Returns:
            Number of entries removed
        """
        pass
    
    @abstractmethod
    def export_all(self) -> List[Dict[str, Any]]:
        """Export all entries as a list of dictionaries."""
        pass
    
    @abstractmethod
    def import_entries(self, entries: List[Dict[str, Any]]) -> int:
        """
        Import entries from a list of dictionaries.
        
        Args:
            entries: List of entry dictionaries
            
        Returns:
            Number of entries imported
        """
        pass


class SQLiteStorage(MemoryStorage):
    """
    SQLite-based memory storage.
    
    Provides persistent storage with efficient querying.
    Thread-safe with connection pooling.
    """
    
    # Default database location
    DEFAULT_DB_PATH = ".agentic_chain/memory.db"
    
    # Schema version for migrations
    SCHEMA_VERSION = 1
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        auto_create: bool = True,
    ):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to the database file. If None, uses default.
            auto_create: Whether to create the database if it doesn't exist.
        """
        if db_path is None:
            home = Path.home()
            db_path = str(home / self.DEFAULT_DB_PATH)
        
        self.db_path = db_path
        self._local = threading.local()
        
        if auto_create:
            self._ensure_db_exists()
            self._ensure_schema()
    
    def _ensure_db_exists(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn
    
    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _ensure_schema(self):
        """Create the database schema if needed."""
        with self._transaction() as cursor:
            # Create schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # Check current version
            cursor.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            current_version = row["version"] if row else 0
            
            if current_version < self.SCHEMA_VERSION:
                self._apply_migrations(cursor, current_version)
    
    def _apply_migrations(self, cursor: sqlite3.Cursor, from_version: int):
        """Apply schema migrations."""
        if from_version < 1:
            # Initial schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    metadata TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    project_path TEXT,
                    tags TEXT,
                    summary TEXT
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type 
                ON memories(memory_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_project 
                ON memories(project_path)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance 
                ON memories(importance DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created 
                ON memories(created_at DESC)
            """)
            
            # Update schema version
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,)
            )
    
    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        with self._transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO memories (
                    id, content, memory_type, metadata, embedding,
                    created_at, updated_at, access_count, importance,
                    project_path, tags, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.content,
                entry.memory_type.value,
                json.dumps(entry.metadata) if entry.metadata else None,
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.created_at,
                entry.updated_at,
                entry.access_count,
                entry.importance,
                entry.project_path,
                json.dumps(entry.tags) if entry.tags else None,
                entry.summary,
            ))
            
        logger.debug(f"Stored memory entry: {entry.id}")
        return entry.id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM memories WHERE id = ?",
                (entry_id,)
            )
            row = cursor.fetchone()
            
            if row:
                entry = self._row_to_entry(row)
                # Update access count
                entry.touch()
                self.update(entry)
                return entry
            
        return None
    
    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []
        
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            metadata=metadata,
            embedding=embedding,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            importance=row["importance"],
            project_path=row["project_path"],
            tags=tags,
            summary=row["summary"],
        )
    
    def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memory entries matching a query."""
        conditions = []
        params = []
        
        # Filter by memory type
        if query.memory_types:
            placeholders = ",".join("?" * len(query.memory_types))
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend([mt.value for mt in query.memory_types])
        
        # Filter by project path
        if query.project_path:
            conditions.append("project_path = ?")
            params.append(query.project_path)
        
        # Filter by minimum importance
        if query.min_importance > 0:
            conditions.append("importance >= ?")
            params.append(query.min_importance)
        
        # Filter by date
        if query.since:
            conditions.append("created_at >= ?")
            params.append(query.since)
        
        # Filter by tags (any match)
        if query.tags:
            tag_conditions = []
            for tag in query.tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")
        
        # Build query
        sql = "SELECT * FROM memories"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY importance DESC, updated_at DESC"
        sql += f" LIMIT {query.limit}"
        
        results = []
        with self._transaction() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            for row in rows:
                entry = self._row_to_entry(row)
                score = self._calculate_score(entry, query)
                results.append(MemorySearchResult(
                    entry=entry,
                    score=score,
                    match_type="semantic" if query.use_semantic else "filter",
                ))
        
        # Sort by score if we have a query text
        if query.query_text and query.use_semantic:
            results = self._rerank_by_text(results, query.query_text)
        
        return results[:query.limit]
    
    def _calculate_score(self, entry: MemoryEntry, query: MemoryQuery) -> float:
        """Calculate relevance score for an entry."""
        score = entry.importance
        
        # Boost for recency
        if entry.updated_at:
            if isinstance(entry.updated_at, str):
                updated = datetime.fromisoformat(entry.updated_at.replace('Z', '+00:00'))
            else:
                updated = entry.updated_at
            now = utc_now()
            # Make both timezone-aware or naive for comparison
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age_days = (now - updated).days
            recency_boost = max(0, 1 - (age_days / 365))
            score = score * 0.7 + recency_boost * 0.3
        
        # Boost for access frequency
        access_boost = min(1.0, entry.access_count / 100)
        score = score * 0.8 + access_boost * 0.2
        
        return score
    
    def _rerank_by_text(
        self,
        results: List[MemorySearchResult],
        query_text: str,
    ) -> List[MemorySearchResult]:
        """Rerank results based on text similarity."""
        query_words = set(query_text.lower().split())
        
        for result in results:
            content_words = set(result.entry.content.lower().split())
            overlap = len(query_words & content_words)
            text_score = overlap / max(len(query_words), 1)
            
            # Combine with existing score
            result.score = result.score * 0.5 + text_score * 0.5
            if text_score > 0:
                result.match_type = "keyword"
        
        # Sort by combined score
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def update(self, entry: MemoryEntry) -> bool:
        """Update an existing memory entry."""
        entry.updated_at = utc_now()
        
        with self._transaction() as cursor:
            cursor.execute("""
                UPDATE memories SET
                    content = ?,
                    memory_type = ?,
                    metadata = ?,
                    embedding = ?,
                    updated_at = ?,
                    access_count = ?,
                    importance = ?,
                    project_path = ?,
                    tags = ?,
                    summary = ?
                WHERE id = ?
            """, (
                entry.content,
                entry.memory_type.value,
                json.dumps(entry.metadata) if entry.metadata else None,
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.updated_at,
                entry.access_count,
                entry.importance,
                entry.project_path,
                json.dumps(entry.tags) if entry.tags else None,
                entry.summary,
                entry.id,
            ))
            
            return cursor.rowcount > 0
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
            deleted = cursor.rowcount > 0
            
        if deleted:
            logger.debug(f"Deleted memory entry: {entry_id}")
        return deleted
    
    def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        stats = MemoryStats()
        
        with self._transaction() as cursor:
            # Total entries
            cursor.execute("SELECT COUNT(*) as count FROM memories")
            stats.total_entries = cursor.fetchone()["count"]
            
            # Entries by type
            cursor.execute("""
                SELECT memory_type, COUNT(*) as count 
                FROM memories 
                GROUP BY memory_type
            """)
            for row in cursor.fetchall():
                stats.entries_by_type[row["memory_type"]] = row["count"]
            
            # Average importance and total access count
            cursor.execute("""
                SELECT 
                    AVG(importance) as avg_importance,
                    SUM(access_count) as total_access
                FROM memories
            """)
            row = cursor.fetchone()
            stats.average_importance = row["avg_importance"] or 0.0
            stats.total_access_count = row["total_access"] or 0
            
            # Date range
            cursor.execute("""
                SELECT MIN(created_at) as oldest, MAX(created_at) as newest
                FROM memories
            """)
            row = cursor.fetchone()
            stats.oldest_entry = row["oldest"]
            stats.newest_entry = row["newest"]
            
            # Database size
            db_path = Path(self.db_path)
            if db_path.exists():
                stats.total_size_bytes = db_path.stat().st_size
        
        return stats
    
    def prune(
        self,
        max_age_days: Optional[int] = None,
        max_entries: Optional[int] = None,
        min_importance: Optional[float] = None,
    ) -> int:
        """Prune old or low-importance entries."""
        total_deleted = 0
        
        with self._transaction() as cursor:
            # Delete by age
            if max_age_days is not None:
                cutoff = utc_now() - timedelta(days=max_age_days)
                # Convert to ISO format string for SQLite comparison
                cutoff_str = cutoff.isoformat()
                cursor.execute(
                    "DELETE FROM memories WHERE created_at < ?",
                    (cutoff_str,)
                )
                total_deleted += cursor.rowcount
            
            # Delete by importance (only short-term memory)
            if min_importance is not None:
                cursor.execute("""
                    DELETE FROM memories 
                    WHERE importance < ? 
                    AND memory_type = ?
                """, (min_importance, MemoryType.SHORT_TERM.value))
                total_deleted += cursor.rowcount
            
            # Keep only top N entries by importance
            if max_entries is not None:
                cursor.execute("SELECT COUNT(*) as count FROM memories")
                current_count = cursor.fetchone()["count"]
                
                if current_count > max_entries:
                    to_delete = current_count - max_entries
                    cursor.execute("""
                        DELETE FROM memories WHERE id IN (
                            SELECT id FROM memories 
                            ORDER BY importance ASC, access_count ASC
                            LIMIT ?
                        )
                    """, (to_delete,))
                    total_deleted += cursor.rowcount
        
        if total_deleted > 0:
            logger.info(f"Pruned {total_deleted} memory entries")
            # Vacuum database to reclaim space
            self._conn.execute("VACUUM")
        
        return total_deleted
    
    def export_all(self) -> List[Dict[str, Any]]:
        """Export all entries as a list of dictionaries."""
        entries = []
        
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM memories ORDER BY created_at")
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                entries.append(entry.to_dict())
        
        return entries
    
    def import_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Import entries from a list of dictionaries."""
        imported = 0
        
        for entry_dict in entries:
            try:
                entry = MemoryEntry.from_dict(entry_dict)
                self.store(entry)
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import entry: {e}")
        
        logger.info(f"Imported {imported} memory entries")
        return imported
    
    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def clear_all(self):
        """Clear all memory entries. Use with caution!"""
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM memories")
            
        self._conn.execute("VACUUM")
        logger.warning("Cleared all memory entries")
