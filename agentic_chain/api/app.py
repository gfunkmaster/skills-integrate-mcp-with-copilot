"""
FastAPI application for Agentic Chain REST API.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..orchestrator import AgenticChain
from .auth import is_auth_enabled, require_api_key
from .jobs import Job, JobStore, job_store
from .logging_middleware import RequestLoggingMiddleware
from .models import (
    AnalyzeIssueRequest,
    AnalyzeProjectRequest,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    WebhookListResponse,
    WebhookRequest,
    WebhookResponse,
)
from .rate_limit import RateLimitMiddleware
from .webhooks import webhook_manager

logger = logging.getLogger(__name__)


# Background task for job cleanup
async def cleanup_old_jobs():
    """Periodically clean up old jobs."""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        job_store.cleanup_old_jobs(max_age_hours=24)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agentic Chain API")
    cleanup_task = asyncio.create_task(cleanup_old_jobs())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic Chain API")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Agentic Chain API",
        description="""
REST API for Agentic Chain - AI-powered GitHub issue analysis and solutions.

## Features
- Analyze GitHub issues with AI-powered insights
- Analyze project structure and patterns
- Async job processing for long-running tasks
- Webhook notifications for job completion
- API key authentication
- Rate limiting

## Authentication
If API keys are configured (via `AGENTIC_CHAIN_API_KEY` or `AGENTIC_CHAIN_API_KEYS` environment variables),
all endpoints except `/api/v1/health` require the `X-API-Key` header.

## Rate Limiting
Requests are rate-limited to prevent abuse. Configure with:
- `AGENTIC_CHAIN_RATE_LIMIT`: Requests per second (default: 10)
- `AGENTIC_CHAIN_RATE_CAPACITY`: Burst capacity (default: 100)
        """,
        version="1.0.0",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("AGENTIC_CHAIN_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI):
    """Register all API routes."""
    
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check endpoint",
    )
    async def health_check():
        """
        Check API health status.
        
        This endpoint does not require authentication.
        """
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
        )
    
    @app.post(
        "/api/v1/issues/analyze",
        response_model=JobResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            422: {"model": ErrorResponse, "description": "Validation Error"},
        },
        tags=["Analysis"],
        summary="Analyze a GitHub issue",
    )
    async def analyze_issue(
        request: AnalyzeIssueRequest,
        background_tasks: BackgroundTasks,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        Start async analysis of a GitHub issue.
        
        Returns a job ID that can be used to check status and get results.
        """
        # Validate project path exists
        project_path = Path(request.project_path)
        if not project_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Project path does not exist: {request.project_path}",
            )
        
        # Create job
        job = job_store.create_job(
            job_type="analyze_issue",
            params={
                "issue_data": {
                    "title": request.title,
                    "body": request.body,
                    "labels": [{"name": label.name} for label in request.labels],
                    "number": request.number,
                },
                "project_path": request.project_path,
                "llm_config": request.llm_config,
                "webhook_url": request.webhook_url,
            },
        )
        
        # Start background processing
        background_tasks.add_task(process_issue_analysis, job.job_id)
        
        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            message="Issue analysis job created",
        )
    
    @app.post(
        "/api/v1/projects/analyze",
        response_model=JobResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            422: {"model": ErrorResponse, "description": "Validation Error"},
        },
        tags=["Analysis"],
        summary="Analyze a project",
    )
    async def analyze_project(
        request: AnalyzeProjectRequest,
        background_tasks: BackgroundTasks,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        Start async analysis of a project.
        
        Returns a job ID that can be used to check status and get results.
        """
        # Validate project path exists
        project_path = Path(request.project_path)
        if not project_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Project path does not exist: {request.project_path}",
            )
        
        # Create job
        job = job_store.create_job(
            job_type="analyze_project",
            params={
                "project_path": request.project_path,
                "llm_config": request.llm_config,
                "webhook_url": request.webhook_url,
            },
        )
        
        # Start background processing
        background_tasks.add_task(process_project_analysis, job.job_id)
        
        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            message="Project analysis job created",
        )
    
    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobStatusResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            404: {"model": ErrorResponse, "description": "Job not found"},
        },
        tags=["Jobs"],
        summary="Get job status",
    )
    async def get_job_status(
        job_id: str,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        Get the status of an analysis job.
        """
        job = job_store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress=job.progress,
            error=job.error,
        )
    
    @app.get(
        "/api/v1/results/{job_id}",
        response_model=JobResultResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            404: {"model": ErrorResponse, "description": "Job not found"},
        },
        tags=["Jobs"],
        summary="Get job results",
    )
    async def get_job_results(
        job_id: str,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        Get the results of a completed analysis job.
        
        Results are only available for completed jobs.
        """
        job = job_store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobResultResponse(
            job_id=job.job_id,
            status=job.status,
            result=job.result,
            error=job.error,
        )
    
    @app.post(
        "/api/v1/webhooks",
        response_model=WebhookResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
        },
        tags=["Webhooks"],
        summary="Register a webhook",
    )
    async def register_webhook(
        request: WebhookRequest,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        Register a webhook for job completion notifications.
        
        Webhooks receive POST requests with event data when jobs complete or fail.
        """
        webhook = webhook_manager.register(
            url=request.url,
            events=request.events,
            secret=request.secret,
        )
        
        return WebhookResponse(
            webhook_id=webhook.webhook_id,
            url=webhook.url,
            events=webhook.events,
            created_at=webhook.created_at,
        )
    
    @app.get(
        "/api/v1/webhooks",
        response_model=WebhookListResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
        },
        tags=["Webhooks"],
        summary="List webhooks",
    )
    async def list_webhooks(
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        List all registered webhooks.
        """
        webhooks = webhook_manager.list_webhooks()
        return WebhookListResponse(
            webhooks=[
                WebhookResponse(
                    webhook_id=w.webhook_id,
                    url=w.url,
                    events=w.events,
                    created_at=w.created_at,
                )
                for w in webhooks
            ]
        )
    
    @app.delete(
        "/api/v1/webhooks/{webhook_id}",
        responses={
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            404: {"model": ErrorResponse, "description": "Webhook not found"},
        },
        tags=["Webhooks"],
        summary="Delete a webhook",
    )
    async def delete_webhook(
        webhook_id: str,
        api_key: Optional[str] = Depends(require_api_key),
    ):
        """
        Delete a registered webhook.
        """
        if not webhook_manager.delete_webhook(webhook_id):
            raise HTTPException(status_code=404, detail="Webhook not found")
        
        return {"message": "Webhook deleted"}


async def process_issue_analysis(job_id: str):
    """Background task to process issue analysis."""
    job = job_store.get_job(job_id)
    if not job:
        return
    
    try:
        await job_store.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress="Starting issue analysis...",
        )
        
        params = job.params
        
        # Create chain and run analysis
        chain = AgenticChain(
            project_path=params["project_path"],
            llm_config=params.get("llm_config"),
        )
        
        await job_store.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress="Running agentic chain...",
        )
        
        # Run synchronous analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            chain.solve_issue,
            params["issue_data"],
        )
        
        await job_store.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result=result,
        )
        
        # Send webhook notification
        await webhook_manager.send_notification(
            "job.completed",
            {"job_id": job_id, "result": result},
            webhook_url=job.webhook_url,
        )
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await job_store.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
        )
        
        # Send webhook notification
        await webhook_manager.send_notification(
            "job.failed",
            {"job_id": job_id, "error": str(e)},
            webhook_url=job.webhook_url,
        )


async def process_project_analysis(job_id: str):
    """Background task to process project analysis."""
    job = job_store.get_job(job_id)
    if not job:
        return
    
    try:
        await job_store.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress="Starting project analysis...",
        )
        
        params = job.params
        
        # Create chain and run analysis
        chain = AgenticChain(
            project_path=params["project_path"],
            llm_config=params.get("llm_config"),
        )
        
        # Run synchronous analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            chain.analyze_project,
        )
        
        await job_store.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result=result,
        )
        
        # Send webhook notification
        await webhook_manager.send_notification(
            "job.completed",
            {"job_id": job_id, "result": result},
            webhook_url=job.webhook_url,
        )
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await job_store.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
        )
        
        # Send webhook notification
        await webhook_manager.send_notification(
            "job.failed",
            {"job_id": job_id, "error": str(e)},
            webhook_url=job.webhook_url,
        )


# Create default app instance
app = create_app()
