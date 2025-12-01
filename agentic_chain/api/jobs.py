"""
In-memory job store for async job processing.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import JobStatus

logger = logging.getLogger(__name__)


class Job:
    """Represents an async job."""
    
    def __init__(self, job_id: str, job_type: str, params: Dict[str, Any]):
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.status = JobStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: Optional[str] = None
        self.webhook_url: Optional[str] = params.get("webhook_url")


class JobStore:
    """In-memory store for jobs."""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> Job:
        """Create a new job."""
        job_id = str(uuid4())
        job = Job(job_id=job_id, job_type=job_type, params=params)
        self._jobs[job_id] = job
        logger.info(f"Created job {job_id} of type {job_type}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        """Update job status."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            job.status = status
            
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now(timezone.utc)
            
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now(timezone.utc)
            
            if progress is not None:
                job.progress = progress
            
            if result is not None:
                job.result = result
            
            if error is not None:
                job.error = error
            
            logger.info(f"Job {job_id} status updated to {status}")
            return job
    
    def list_jobs(self, limit: int = 100) -> List[Job]:
        """List recent jobs."""
        # Take a snapshot of jobs to avoid iteration issues
        jobs_snapshot = list(self._jobs.values())
        jobs_sorted = sorted(
            jobs_snapshot,
            key=lambda j: j.created_at,
            reverse=True
        )
        return jobs_sorted[:limit]
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove jobs older than max_age_hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        # Take a snapshot to avoid modification during iteration
        jobs_snapshot = list(self._jobs.items())
        
        to_remove = [
            job_id for job_id, job in jobs_snapshot
            if job.created_at < cutoff
        ]
        
        for job_id in to_remove:
            self._jobs.pop(job_id, None)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")
        
        return len(to_remove)
    
    def clear(self):
        """Clear all jobs. Primarily for testing."""
        self._jobs.clear()


# Global job store instance
job_store = JobStore()
