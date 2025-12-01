"""
Tests for the Memory System.
"""

import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentic_chain.memory import (
    MemoryManager,
    MemoryType,
    MemoryEntry,
    MemoryQuery,
    MemorySearchResult,
    SQLiteStorage,
    SimpleEmbedding,
)
from agentic_chain.memory.types import utc_now


class TestMemoryTypes:
    """Test memory type definitions."""
    
    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.LONG_TERM,
        )
        
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.LONG_TERM
        assert entry.id is not None
        assert entry.created_at is not None
        assert entry.importance == 0.5
        assert entry.access_count == 0
    
    def test_memory_entry_with_metadata(self):
        """Test memory entry with metadata."""
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.ISSUE,
            metadata={"issue_id": 123, "priority": "high"},
            tags=["bug", "critical"],
            importance=0.9,
        )
        
        assert entry.metadata["issue_id"] == 123
        assert "bug" in entry.tags
        assert entry.importance == 0.9
    
    def test_memory_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.PROJECT,
            metadata={"key": "value"},
        )
        
        data = entry.to_dict()
        
        assert data["content"] == "Test content"
        assert data["memory_type"] == "project"
        assert data["metadata"]["key"] == "value"
        assert data["id"] == entry.id
    
    def test_memory_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "id": "test_id",
            "content": "Test content",
            "memory_type": "long_term",
            "metadata": {"key": "value"},
            "importance": 0.7,
            "tags": ["test"],
            "created_at": datetime.utcnow().isoformat(),
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert entry.id == "test_id"
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.LONG_TERM
        assert entry.importance == 0.7
    
    def test_memory_entry_touch(self):
        """Test updating access count."""
        entry = MemoryEntry(
            content="Test",
            memory_type=MemoryType.SHORT_TERM,
        )
        initial_count = entry.access_count
        initial_time = entry.updated_at
        
        entry.touch()
        
        assert entry.access_count == initial_count + 1
        assert entry.updated_at >= initial_time
    
    def test_memory_query_creation(self):
        """Test creating a memory query."""
        query = MemoryQuery(
            query_text="test search",
            memory_types=[MemoryType.ISSUE, MemoryType.PROJECT],
            project_path="/path/to/project",
            limit=5,
        )
        
        assert query.query_text == "test search"
        assert len(query.memory_types) == 2
        assert query.limit == 5


class TestSimpleEmbedding:
    """Test the simple embedding provider."""
    
    def test_embed_text(self):
        """Test embedding text."""
        provider = SimpleEmbedding()
        embedding = provider.embed("Hello world test")
        
        assert len(embedding) == provider.dimension
        assert all(isinstance(v, float) for v in embedding)
    
    def test_embed_empty(self):
        """Test embedding empty text."""
        provider = SimpleEmbedding()
        embedding = provider.embed("")
        
        assert len(embedding) == provider.dimension
        assert all(v == 0.0 for v in embedding)
    
    def test_embed_batch(self):
        """Test batch embedding."""
        provider = SimpleEmbedding()
        texts = ["Hello world", "Test text", "Another example"]
        embeddings = provider.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == provider.dimension for e in embeddings)
    
    def test_similarity_identical(self):
        """Test similarity of identical texts."""
        provider = SimpleEmbedding()
        vec1 = provider.embed("Hello world")
        vec2 = provider.embed("Hello world")
        
        similarity = provider.similarity(vec1, vec2)
        
        assert similarity == 1.0
    
    def test_similarity_different(self):
        """Test similarity of different texts."""
        provider = SimpleEmbedding()
        vec1 = provider.embed("Python programming language")
        vec2 = provider.embed("JavaScript web development")
        
        similarity = provider.similarity(vec1, vec2)
        
        assert 0.0 <= similarity <= 1.0
    
    def test_similarity_related(self):
        """Test similarity of related texts."""
        provider = SimpleEmbedding()
        vec1 = provider.embed("login authentication bug")
        vec2 = provider.embed("authentication login error")
        
        similarity = provider.similarity(vec1, vec2)
        
        # Related texts should have higher similarity
        assert similarity > 0.5


class TestSQLiteStorage:
    """Test SQLite storage backend."""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create a temporary storage instance."""
        db_path = str(tmp_path / "test_memory.db")
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        storage.close()
    
    def test_store_and_retrieve(self, storage):
        """Test storing and retrieving an entry."""
        entry = MemoryEntry(
            content="Test memory content",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8,
        )
        
        entry_id = storage.store(entry)
        
        retrieved = storage.retrieve(entry_id)
        
        assert retrieved is not None
        assert retrieved.content == "Test memory content"
        assert retrieved.memory_type == MemoryType.LONG_TERM
        assert retrieved.importance == 0.8
    
    def test_retrieve_updates_access_count(self, storage):
        """Test that retrieval updates access count."""
        entry = MemoryEntry(
            content="Test",
            memory_type=MemoryType.SHORT_TERM,
        )
        
        entry_id = storage.store(entry)
        
        retrieved1 = storage.retrieve(entry_id)
        assert retrieved1.access_count == 1
        
        retrieved2 = storage.retrieve(entry_id)
        assert retrieved2.access_count == 2
    
    def test_retrieve_not_found(self, storage):
        """Test retrieving non-existent entry."""
        result = storage.retrieve("nonexistent_id")
        assert result is None
    
    def test_search_by_type(self, storage):
        """Test searching by memory type."""
        # Store entries of different types
        storage.store(MemoryEntry(
            content="Project analysis",
            memory_type=MemoryType.PROJECT,
        ))
        storage.store(MemoryEntry(
            content="Issue solution",
            memory_type=MemoryType.ISSUE,
        ))
        storage.store(MemoryEntry(
            content="Another project",
            memory_type=MemoryType.PROJECT,
        ))
        
        query = MemoryQuery(
            memory_types=[MemoryType.PROJECT],
            limit=10,
        )
        
        results = storage.search(query)
        
        assert len(results) == 2
        assert all(r.entry.memory_type == MemoryType.PROJECT for r in results)
    
    def test_search_by_project_path(self, storage):
        """Test searching by project path."""
        storage.store(MemoryEntry(
            content="Project A data",
            memory_type=MemoryType.PROJECT,
            project_path="/path/to/project_a",
        ))
        storage.store(MemoryEntry(
            content="Project B data",
            memory_type=MemoryType.PROJECT,
            project_path="/path/to/project_b",
        ))
        
        query = MemoryQuery(
            project_path="/path/to/project_a",
            limit=10,
        )
        
        results = storage.search(query)
        
        assert len(results) == 1
        assert results[0].entry.project_path == "/path/to/project_a"
    
    def test_search_by_tags(self, storage):
        """Test searching by tags."""
        storage.store(MemoryEntry(
            content="Bug fix",
            memory_type=MemoryType.ISSUE,
            tags=["bug", "fix"],
        ))
        storage.store(MemoryEntry(
            content="Feature request",
            memory_type=MemoryType.ISSUE,
            tags=["feature", "enhancement"],
        ))
        
        query = MemoryQuery(
            tags=["bug"],
            limit=10,
        )
        
        results = storage.search(query)
        
        assert len(results) == 1
        assert "bug" in results[0].entry.tags
    
    def test_update_entry(self, storage):
        """Test updating an entry."""
        entry = MemoryEntry(
            content="Original content",
            memory_type=MemoryType.LONG_TERM,
        )
        
        entry_id = storage.store(entry)
        
        # Update the entry
        entry.content = "Updated content"
        entry.importance = 0.9
        storage.update(entry)
        
        retrieved = storage.retrieve(entry_id)
        
        assert retrieved.content == "Updated content"
        assert retrieved.importance == 0.9
    
    def test_delete_entry(self, storage):
        """Test deleting an entry."""
        entry = MemoryEntry(
            content="To be deleted",
            memory_type=MemoryType.SHORT_TERM,
        )
        
        entry_id = storage.store(entry)
        
        # Verify it exists
        assert storage.retrieve(entry_id) is not None
        
        # Delete
        result = storage.delete(entry_id)
        
        assert result is True
        assert storage.retrieve(entry_id) is None
    
    def test_delete_not_found(self, storage):
        """Test deleting non-existent entry."""
        result = storage.delete("nonexistent_id")
        assert result is False
    
    def test_get_stats(self, storage):
        """Test getting storage statistics."""
        # Store some entries
        storage.store(MemoryEntry(
            content="Entry 1",
            memory_type=MemoryType.PROJECT,
            importance=0.8,
        ))
        storage.store(MemoryEntry(
            content="Entry 2",
            memory_type=MemoryType.ISSUE,
            importance=0.6,
        ))
        storage.store(MemoryEntry(
            content="Entry 3",
            memory_type=MemoryType.PROJECT,
            importance=0.7,
        ))
        
        stats = storage.get_stats()
        
        assert stats.total_entries == 3
        assert stats.entries_by_type["project"] == 2
        assert stats.entries_by_type["issue"] == 1
        assert stats.average_importance == pytest.approx(0.7, rel=0.1)
    
    def test_prune_by_age(self, storage):
        """Test pruning old entries."""
        # Store entries with different ages
        old_entry = MemoryEntry(
            content="Old entry",
            memory_type=MemoryType.SHORT_TERM,
        )
        old_entry.created_at = utc_now() - timedelta(days=30)
        storage.store(old_entry)
        
        new_entry = MemoryEntry(
            content="New entry",
            memory_type=MemoryType.SHORT_TERM,
        )
        storage.store(new_entry)
        
        # Prune entries older than 7 days
        deleted = storage.prune(max_age_days=7)
        
        assert deleted == 1
        
        # Check that only new entry remains
        stats = storage.get_stats()
        assert stats.total_entries == 1
    
    def test_prune_by_max_entries(self, storage):
        """Test pruning to keep max entries."""
        # Store multiple entries
        for i in range(5):
            storage.store(MemoryEntry(
                content=f"Entry {i}",
                memory_type=MemoryType.LONG_TERM,
                importance=0.1 * (i + 1),  # Varying importance
            ))
        
        # Keep only 3 entries
        deleted = storage.prune(max_entries=3)
        
        assert deleted == 2
        
        stats = storage.get_stats()
        assert stats.total_entries == 3
    
    def test_export_import(self, storage, tmp_path):
        """Test exporting and importing entries."""
        # Store entries
        storage.store(MemoryEntry(
            content="Entry 1",
            memory_type=MemoryType.PROJECT,
        ))
        storage.store(MemoryEntry(
            content="Entry 2",
            memory_type=MemoryType.ISSUE,
        ))
        
        # Export
        entries = storage.export_all()
        
        assert len(entries) == 2
        
        # Create new storage and import
        new_storage = SQLiteStorage(db_path=str(tmp_path / "new_memory.db"))
        imported = new_storage.import_entries(entries)
        
        assert imported == 2
        assert new_storage.get_stats().total_entries == 2
        
        new_storage.close()


class TestMemoryManager:
    """Test the high-level MemoryManager."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a temporary memory manager."""
        db_path = str(tmp_path / "test_memory.db")
        manager = MemoryManager(db_path=db_path)
        yield manager
        manager.close()
    
    def test_remember_and_recall(self, manager):
        """Test basic remember and recall."""
        memory_id = manager.remember(
            content="Important information",
            memory_type=MemoryType.LONG_TERM,
        )
        
        entry = manager.recall(memory_id)
        
        assert entry is not None
        assert entry.content == "Important information"
    
    def test_remember_with_metadata(self, manager):
        """Test remember with metadata and tags."""
        memory_id = manager.remember(
            content="Project analysis",
            memory_type=MemoryType.PROJECT,
            metadata={"languages": ["Python", "JavaScript"]},
            tags=["analysis", "project"],
            project_path="/path/to/project",
            importance=0.9,
        )
        
        entry = manager.recall(memory_id)
        
        assert entry.metadata["languages"] == ["Python", "JavaScript"]
        assert "analysis" in entry.tags
        assert entry.project_path == "/path/to/project"
    
    def test_search(self, manager):
        """Test searching memories."""
        manager.remember(
            content="Login authentication bug fix",
            memory_type=MemoryType.ISSUE,
            tags=["bug", "auth"],
        )
        manager.remember(
            content="Feature enhancement request",
            memory_type=MemoryType.ISSUE,
            tags=["feature"],
        )
        manager.remember(
            content="Project structure analysis",
            memory_type=MemoryType.PROJECT,
        )
        
        results = manager.search(
            query="authentication",
            memory_types=[MemoryType.ISSUE],
        )
        
        assert len(results) >= 1
        assert any("authentication" in r.entry.content for r in results)
    
    def test_forget(self, manager):
        """Test forgetting a memory."""
        memory_id = manager.remember(
            content="Temporary data",
            memory_type=MemoryType.SHORT_TERM,
        )
        
        result = manager.forget(memory_id)
        
        assert result is True
        assert manager.recall(memory_id) is None
    
    def test_store_project_analysis(self, manager):
        """Test storing project analysis."""
        analysis = {
            "languages": {"Python": 10, "JavaScript": 5},
            "patterns": {"framework": "FastAPI"},
            "structure": {"file_count": 15},
        }
        
        memory_id = manager.store_project_analysis(
            project_path="/path/to/project",
            analysis=analysis,
        )
        
        entry = manager.recall(memory_id)
        
        assert entry is not None
        assert entry.memory_type == MemoryType.PROJECT
        assert "project_analysis" in entry.tags
    
    def test_store_issue_solution(self, manager):
        """Test storing issue solution."""
        solution = {
            "issue_type": "bug",
            "proposed_changes": [{"file": "main.py", "change": "fix"}],
        }
        
        memory_id = manager.store_issue_solution(
            issue_title="Login bug",
            issue_body="Users cannot login",
            solution=solution,
            project_path="/project",
        )
        
        entry = manager.recall(memory_id)
        
        assert entry is not None
        assert entry.memory_type == MemoryType.ISSUE
        assert "Login bug" in entry.content
    
    def test_get_relevant_context(self, manager):
        """Test getting relevant context."""
        manager.remember(
            content="Authentication module handles login",
            memory_type=MemoryType.ENTITY,
            project_path="/project",
        )
        manager.remember(
            content="Database connection pooling",
            memory_type=MemoryType.ENTITY,
            project_path="/project",
        )
        
        context = manager.get_relevant_context(
            query="login authentication",
            project_path="/project",
        )
        
        assert len(context) >= 1
    
    def test_get_similar_issues(self, manager):
        """Test finding similar issues."""
        manager.store_issue_solution(
            issue_title="Login not working",
            issue_body="Users see error on login page",
            solution={"fix": "Update auth module"},
        )
        manager.store_issue_solution(
            issue_title="Database timeout",
            issue_body="Queries taking too long",
            solution={"fix": "Add indexes"},
        )
        
        similar = manager.get_similar_issues(
            issue_title="Authentication error",
            issue_body="Login fails with error",
        )
        
        assert len(similar) >= 1
    
    def test_export_import(self, manager, tmp_path):
        """Test export and import functionality."""
        manager.remember("Memory 1", MemoryType.LONG_TERM)
        manager.remember("Memory 2", MemoryType.PROJECT)
        
        export_path = str(tmp_path / "export.json")
        
        exported = manager.export_memories(export_path)
        
        assert exported == 2
        assert Path(export_path).exists()
        
        # Verify export file contents
        with open(export_path) as f:
            data = json.load(f)
        
        assert "entries" in data
        assert len(data["entries"]) == 2
        
        # Create new manager and import
        new_manager = MemoryManager(
            db_path=str(tmp_path / "new_memory.db")
        )
        imported = new_manager.import_memories(export_path)
        
        assert imported == 2
        new_manager.close()
    
    def test_get_stats(self, manager):
        """Test getting statistics."""
        manager.remember("Entry 1", MemoryType.PROJECT)
        manager.remember("Entry 2", MemoryType.ISSUE)
        manager.remember("Entry 3", MemoryType.ISSUE)
        
        stats = manager.get_stats()
        
        assert stats.total_entries == 3
        assert stats.entries_by_type.get("project") == 1
        assert stats.entries_by_type.get("issue") == 2
    
    def test_get_stats_summary(self, manager):
        """Test getting stats summary string."""
        manager.remember("Test", MemoryType.LONG_TERM)
        
        summary = manager.get_stats_summary()
        
        assert "Memory System Statistics" in summary
        assert "Total entries: 1" in summary
    
    def test_prune(self, manager):
        """Test pruning memories."""
        # Create old entries
        for i in range(5):
            entry = MemoryEntry(
                content=f"Entry {i}",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.1 * (i + 1),
            )
            entry.created_at = utc_now() - timedelta(days=30)
            manager._storage.store(entry)
        
        # Add new entry
        manager.remember("New entry", MemoryType.LONG_TERM)
        
        # Prune old entries
        deleted = manager.prune(max_age_days=7)
        
        assert deleted == 5
        assert manager.get_stats().total_entries == 1
    
    def test_semantic_search(self, manager):
        """Test semantic search with embeddings."""
        manager.remember(
            content="Python FastAPI web development framework",
            memory_type=MemoryType.PROJECT,
        )
        manager.remember(
            content="JavaScript React frontend library",
            memory_type=MemoryType.PROJECT,
        )
        
        results = manager.search("web API development")
        
        # Should find FastAPI-related entry with higher score
        assert len(results) >= 1


class TestMemoryPerformance:
    """Test memory system performance requirements."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a temporary memory manager."""
        db_path = str(tmp_path / "perf_test.db")
        manager = MemoryManager(db_path=db_path)
        yield manager
        manager.close()
    
    def test_retrieval_speed(self, manager):
        """Test that retrieval is fast (< 100ms)."""
        import time
        
        # Store some entries
        for i in range(100):
            manager.remember(
                content=f"Memory entry {i} with some content",
                memory_type=MemoryType.LONG_TERM,
            )
        
        # Time retrieval
        start = time.perf_counter()
        results = manager.search("memory content", limit=10)
        elapsed = time.perf_counter() - start
        
        # Should be under 100ms
        assert elapsed < 0.1, f"Retrieval took {elapsed:.3f}s, expected < 0.1s"
        assert len(results) > 0
    
    def test_storage_efficiency(self, manager, tmp_path):
        """Test that storage is efficient (< 10MB for typical project)."""
        # Simulate typical project memories
        for i in range(100):
            manager.store_project_analysis(
                project_path=f"/project_{i}",
                analysis={
                    "languages": {"Python": 10},
                    "structure": {"file_count": 50},
                },
            )
        
        for i in range(200):
            manager.store_issue_solution(
                issue_title=f"Issue {i}",
                issue_body=f"Description of issue {i}",
                solution={"fix": "Some solution"},
            )
        
        stats = manager.get_stats()
        size_mb = stats.total_size_bytes / (1024 * 1024)
        
        # Should be under 10MB
        assert size_mb < 10, f"Storage size is {size_mb:.2f}MB, expected < 10MB"
