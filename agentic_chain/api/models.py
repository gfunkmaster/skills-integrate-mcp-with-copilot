"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IssueLabel(BaseModel):
    """Issue label representation."""
    name: str


class AnalyzeIssueRequest(BaseModel):
    """Request model for issue analysis."""
    title: str = Field(..., description="Issue title")
    body: str = Field(default="", description="Issue description/body")
    labels: List[IssueLabel] = Field(default_factory=list, description="Issue labels")
    number: Optional[int] = Field(default=None, description="Issue number")
    project_path: str = Field(..., description="Path to the project to analyze")
    llm_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional LLM configuration (provider, model, api_key)"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL for completion notification"
    )


class AnalyzeProjectRequest(BaseModel):
    """Request model for project analysis."""
    project_path: str = Field(..., description="Path to the project to analyze")
    llm_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional LLM configuration"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL for completion notification"
    )


class JobResponse(BaseModel):
    """Response model for job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(default="Job created successfully", description="Status message")


class JobStatusResponse(BaseModel):
    """Response model for job status query."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")
    progress: Optional[str] = Field(default=None, description="Current progress description")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class JobResultResponse(BaseModel):
    """Response model for job results."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Analysis results (available when completed)"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class WebhookRequest(BaseModel):
    """Request model for webhook registration."""
    url: str = Field(..., description="Webhook URL to receive notifications")
    events: List[str] = Field(
        default_factory=lambda: ["job.completed", "job.failed"],
        description="Events to subscribe to"
    )
    secret: Optional[str] = Field(
        default=None,
        description="Optional secret for webhook signature verification"
    )


class WebhookResponse(BaseModel):
    """Response model for webhook registration."""
    webhook_id: str = Field(..., description="Unique webhook identifier")
    url: str = Field(..., description="Registered webhook URL")
    events: List[str] = Field(..., description="Subscribed events")
    created_at: datetime = Field(..., description="Registration timestamp")


class WebhookListResponse(BaseModel):
    """Response model for listing webhooks."""
    webhooks: List[WebhookResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
