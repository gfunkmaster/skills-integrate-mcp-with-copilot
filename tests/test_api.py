"""Tests for the REST API."""

import json
import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from agentic_chain.api.app import create_app, process_issue_analysis, process_project_analysis
from agentic_chain.api.jobs import JobStore, job_store
from agentic_chain.api.models import JobStatus
from agentic_chain.api.webhooks import WebhookManager, webhook_manager
from agentic_chain.api.auth import verify_api_key, get_api_keys, is_auth_enabled


@pytest.fixture
def test_app():
    """Create test app instance."""
    return create_app()


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def clear_stores():
    """Clear job and webhook stores before each test."""
    job_store.clear()
    webhook_manager.clear()
    yield


@pytest.fixture
def sample_project_path(tmp_path):
    """Create a sample project directory."""
    (tmp_path / "app.py").write_text("print('hello')")
    (tmp_path / "README.md").write_text("# Test Project")
    return str(tmp_path)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_health_check_no_auth_required(self, client):
        """Test health endpoint doesn't require auth even when enabled."""
        with patch.dict(os.environ, {"AGENTIC_CHAIN_API_KEY": "test-key"}):
            response = client.get("/api/v1/health")
            assert response.status_code == 200


class TestIssueAnalysisEndpoint:
    """Test issue analysis endpoint."""
    
    def test_analyze_issue_creates_job(self, client, sample_project_path):
        """Test that analyze issue creates a job."""
        response = client.post(
            "/api/v1/issues/analyze",
            json={
                "title": "Test Issue",
                "body": "This is a test issue",
                "labels": [{"name": "bug"}],
                "project_path": sample_project_path,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "created_at" in data
    
    def test_analyze_issue_invalid_project_path(self, client):
        """Test that invalid project path returns error."""
        response = client.post(
            "/api/v1/issues/analyze",
            json={
                "title": "Test Issue",
                "body": "This is a test issue",
                "project_path": "/nonexistent/path",
            },
        )
        
        assert response.status_code == 422
        assert "does not exist" in response.json()["detail"]
    
    def test_analyze_issue_missing_required_fields(self, client):
        """Test that missing required fields returns validation error."""
        response = client.post(
            "/api/v1/issues/analyze",
            json={
                "body": "Missing title and project_path",
            },
        )
        
        assert response.status_code == 422


class TestProjectAnalysisEndpoint:
    """Test project analysis endpoint."""
    
    def test_analyze_project_creates_job(self, client, sample_project_path):
        """Test that analyze project creates a job."""
        response = client.post(
            "/api/v1/projects/analyze",
            json={
                "project_path": sample_project_path,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
    
    def test_analyze_project_invalid_path(self, client):
        """Test that invalid project path returns error."""
        response = client.post(
            "/api/v1/projects/analyze",
            json={
                "project_path": "/nonexistent/path",
            },
        )
        
        assert response.status_code == 422


class TestJobStatusEndpoint:
    """Test job status endpoint."""
    
    def test_get_job_status(self, client, sample_project_path):
        """Test getting job status."""
        # Create a job first
        create_response = client.post(
            "/api/v1/projects/analyze",
            json={"project_path": sample_project_path},
        )
        job_id = create_response.json()["job_id"]
        
        # Get status
        response = client.get(f"/api/v1/jobs/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "created_at" in data
    
    def test_get_job_status_not_found(self, client):
        """Test getting status of nonexistent job."""
        response = client.get("/api/v1/jobs/nonexistent-job-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestJobResultsEndpoint:
    """Test job results endpoint."""
    
    def test_get_job_results(self, client, sample_project_path):
        """Test getting job results."""
        # Create a job
        create_response = client.post(
            "/api/v1/projects/analyze",
            json={"project_path": sample_project_path},
        )
        job_id = create_response.json()["job_id"]
        
        # Get results
        response = client.get(f"/api/v1/results/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
    
    def test_get_job_results_not_found(self, client):
        """Test getting results of nonexistent job."""
        response = client.get("/api/v1/results/nonexistent-job-id")
        
        assert response.status_code == 404


class TestWebhookEndpoints:
    """Test webhook endpoints."""
    
    def test_register_webhook(self, client):
        """Test webhook registration."""
        response = client.post(
            "/api/v1/webhooks",
            json={
                "url": "https://example.com/webhook",
                "events": ["job.completed", "job.failed"],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "webhook_id" in data
        assert data["url"] == "https://example.com/webhook"
        assert "job.completed" in data["events"]
    
    def test_list_webhooks(self, client):
        """Test listing webhooks."""
        # Register a webhook first
        client.post(
            "/api/v1/webhooks",
            json={"url": "https://example.com/webhook"},
        )
        
        response = client.get("/api/v1/webhooks")
        
        assert response.status_code == 200
        data = response.json()
        assert "webhooks" in data
        assert len(data["webhooks"]) == 1
    
    def test_delete_webhook(self, client):
        """Test deleting a webhook."""
        # Register a webhook first
        create_response = client.post(
            "/api/v1/webhooks",
            json={"url": "https://example.com/webhook"},
        )
        webhook_id = create_response.json()["webhook_id"]
        
        # Delete it
        response = client.delete(f"/api/v1/webhooks/{webhook_id}")
        
        assert response.status_code == 200
        
        # Verify it's gone
        list_response = client.get("/api/v1/webhooks")
        assert len(list_response.json()["webhooks"]) == 0
    
    def test_delete_webhook_not_found(self, client):
        """Test deleting nonexistent webhook."""
        response = client.delete("/api/v1/webhooks/nonexistent-id")
        
        assert response.status_code == 404


class TestAuthentication:
    """Test API authentication."""
    
    def test_auth_disabled_when_no_keys(self):
        """Test auth is disabled when no keys configured."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            os.environ.pop("AGENTIC_CHAIN_API_KEY", None)
            os.environ.pop("AGENTIC_CHAIN_API_KEYS", None)
            
            assert not is_auth_enabled()
    
    def test_auth_enabled_with_single_key(self):
        """Test auth is enabled with single key."""
        with patch.dict(os.environ, {"AGENTIC_CHAIN_API_KEY": "test-key"}):
            assert is_auth_enabled()
            assert verify_api_key("test-key")
            assert not verify_api_key("wrong-key")
    
    def test_auth_enabled_with_multiple_keys(self):
        """Test auth is enabled with multiple keys."""
        with patch.dict(os.environ, {"AGENTIC_CHAIN_API_KEYS": "key1,key2,key3"}):
            assert is_auth_enabled()
            assert verify_api_key("key1")
            assert verify_api_key("key2")
            assert verify_api_key("key3")
            assert not verify_api_key("key4")
    
    def test_endpoint_requires_auth_when_enabled(self, client, sample_project_path):
        """Test endpoints require auth when enabled."""
        with patch.dict(os.environ, {"AGENTIC_CHAIN_API_KEY": "test-key"}):
            # Without API key
            response = client.post(
                "/api/v1/projects/analyze",
                json={"project_path": sample_project_path},
            )
            assert response.status_code == 401
            
            # With correct API key
            response = client.post(
                "/api/v1/projects/analyze",
                json={"project_path": sample_project_path},
                headers={"X-API-Key": "test-key"},
            )
            assert response.status_code == 200


class TestJobStore:
    """Test job store functionality."""
    
    def test_create_job(self):
        """Test job creation."""
        store = JobStore()
        job = store.create_job("test", {"param": "value"})
        
        assert job.job_id is not None
        assert job.status == JobStatus.PENDING
        assert job.params == {"param": "value"}
    
    def test_get_job(self):
        """Test getting a job."""
        store = JobStore()
        job = store.create_job("test", {})
        
        retrieved = store.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id
    
    def test_get_nonexistent_job(self):
        """Test getting nonexistent job returns None."""
        store = JobStore()
        assert store.get_job("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_update_job_status(self):
        """Test updating job status."""
        store = JobStore()
        job = store.create_job("test", {})
        
        updated = await store.update_job_status(
            job.job_id,
            JobStatus.RUNNING,
            progress="Running...",
        )
        
        assert updated is not None
        assert updated.status == JobStatus.RUNNING
        assert updated.started_at is not None
        assert updated.progress == "Running..."
    
    @pytest.mark.asyncio
    async def test_update_job_completed(self):
        """Test updating job to completed."""
        store = JobStore()
        job = store.create_job("test", {})
        
        updated = await store.update_job_status(
            job.job_id,
            JobStatus.COMPLETED,
            result={"key": "value"},
        )
        
        assert updated.status == JobStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.result == {"key": "value"}


class TestWebhookManager:
    """Test webhook manager functionality."""
    
    def test_register_webhook(self):
        """Test webhook registration."""
        manager = WebhookManager()
        webhook = manager.register(
            url="https://example.com/hook",
            events=["job.completed"],
            secret="secret123",
        )
        
        assert webhook.webhook_id is not None
        assert webhook.url == "https://example.com/hook"
        assert webhook.events == ["job.completed"]
    
    def test_delete_webhook(self):
        """Test webhook deletion."""
        manager = WebhookManager()
        webhook = manager.register(url="https://example.com/hook", events=[])
        
        assert manager.delete_webhook(webhook.webhook_id)
        assert manager.get_webhook(webhook.webhook_id) is None
    
    def test_list_webhooks(self):
        """Test listing webhooks."""
        manager = WebhookManager()
        manager.register(url="https://example1.com/hook", events=[])
        manager.register(url="https://example2.com/hook", events=[])
        
        webhooks = manager.list_webhooks()
        assert len(webhooks) == 2


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation is available."""
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON is available."""
        response = client.get("/api/v1/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Agentic Chain API"
        assert data["info"]["version"] == "1.0.0"
    
    def test_swagger_ui(self, client):
        """Test Swagger UI is available."""
        response = client.get("/api/v1/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
    
    def test_redoc(self, client):
        """Test ReDoc is available."""
        response = client.get("/api/v1/redoc")
        
        assert response.status_code == 200


class TestRequestIdHeader:
    """Test request ID header is added to responses."""
    
    def test_request_id_in_response(self, client):
        """Test that X-Request-ID header is in responses."""
        response = client.get("/api/v1/health")
        
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
class TestBackgroundProcessing:
    """Test background job processing."""
    
    async def test_process_project_analysis(self, sample_project_path):
        """Test project analysis background processing."""
        job = job_store.create_job(
            job_type="analyze_project",
            params={"project_path": sample_project_path},
        )
        
        await process_project_analysis(job.job_id)
        
        updated_job = job_store.get_job(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.result is not None
    
    async def test_process_issue_analysis(self, sample_project_path):
        """Test issue analysis background processing."""
        job = job_store.create_job(
            job_type="analyze_issue",
            params={
                "project_path": sample_project_path,
                "issue_data": {
                    "title": "Test Issue",
                    "body": "Test body",
                    "labels": [],
                },
            },
        )
        
        await process_issue_analysis(job.job_id)
        
        updated_job = job_store.get_job(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.result is not None
    
    async def test_process_job_failure(self):
        """Test job failure handling."""
        job = job_store.create_job(
            job_type="analyze_project",
            params={"project_path": "/nonexistent/path"},
        )
        
        await process_project_analysis(job.job_id)
        
        updated_job = job_store.get_job(job.job_id)
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error is not None
