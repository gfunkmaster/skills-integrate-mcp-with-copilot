"""
High School Management System API

A super simple FastAPI application that allows students to view and sign up
for extracurricular activities at Mergington High School.

Also includes observability API endpoints for the Agentic Chain dashboard.
"""

from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os
from pathlib import Path
from typing import Optional

app = FastAPI(title="Mergington High School API",
              description="API for viewing and signing up for extracurricular activities")

# Mount the static files directory
current_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=os.path.join(Path(__file__).parent,
          "static")), name="static")

# In-memory activity database
activities = {
    "Chess Club": {
        "description": "Learn strategies and compete in chess tournaments",
        "schedule": "Fridays, 3:30 PM - 5:00 PM",
        "max_participants": 12,
        "participants": ["michael@mergington.edu", "daniel@mergington.edu"]
    },
    "Programming Class": {
        "description": "Learn programming fundamentals and build software projects",
        "schedule": "Tuesdays and Thursdays, 3:30 PM - 4:30 PM",
        "max_participants": 20,
        "participants": ["emma@mergington.edu", "sophia@mergington.edu"]
    },
    "Gym Class": {
        "description": "Physical education and sports activities",
        "schedule": "Mondays, Wednesdays, Fridays, 2:00 PM - 3:00 PM",
        "max_participants": 30,
        "participants": ["john@mergington.edu", "olivia@mergington.edu"]
    },
    "Soccer Team": {
        "description": "Join the school soccer team and compete in matches",
        "schedule": "Tuesdays and Thursdays, 4:00 PM - 5:30 PM",
        "max_participants": 22,
        "participants": ["liam@mergington.edu", "noah@mergington.edu"]
    },
    "Basketball Team": {
        "description": "Practice and play basketball with the school team",
        "schedule": "Wednesdays and Fridays, 3:30 PM - 5:00 PM",
        "max_participants": 15,
        "participants": ["ava@mergington.edu", "mia@mergington.edu"]
    },
    "Art Club": {
        "description": "Explore your creativity through painting and drawing",
        "schedule": "Thursdays, 3:30 PM - 5:00 PM",
        "max_participants": 15,
        "participants": ["amelia@mergington.edu", "harper@mergington.edu"]
    },
    "Drama Club": {
        "description": "Act, direct, and produce plays and performances",
        "schedule": "Mondays and Wednesdays, 4:00 PM - 5:30 PM",
        "max_participants": 20,
        "participants": ["ella@mergington.edu", "scarlett@mergington.edu"]
    },
    "Math Club": {
        "description": "Solve challenging problems and participate in math competitions",
        "schedule": "Tuesdays, 3:30 PM - 4:30 PM",
        "max_participants": 10,
        "participants": ["james@mergington.edu", "benjamin@mergington.edu"]
    },
    "Debate Team": {
        "description": "Develop public speaking and argumentation skills",
        "schedule": "Fridays, 4:00 PM - 5:30 PM",
        "max_participants": 12,
        "participants": ["charlotte@mergington.edu", "henry@mergington.edu"]
    }
}

# In-memory observability data storage
_observability_data = {
    "summary": {
        "total_executions": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "success_rate": 0.0,
        "avg_execution_time_ms": 0.0,
        "p95_execution_time_ms": 0.0,
    },
    "llm_usage": {
        "total_tokens": 0,
        "total_cost": 0.0,
    },
    "recent_timelines": [],
    "metrics": {},
    "recent_errors": [],
    "generated_at": None,
}


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/dashboard")
def dashboard():
    """Redirect to the observability dashboard."""
    return RedirectResponse(url="/static/dashboard.html")


@app.get("/activities")
def get_activities():
    return activities


@app.post("/activities/{activity_name}/signup")
def signup_for_activity(activity_name: str, email: str):
    """Sign up a student for an activity"""
    # Validate activity exists
    if activity_name not in activities:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Get the specific activity
    activity = activities[activity_name]

    # Validate student is not already signed up
    if email in activity["participants"]:
        raise HTTPException(
            status_code=400,
            detail="Student is already signed up"
        )

    # Add student
    activity["participants"].append(email)
    return {"message": f"Signed up {email} for {activity_name}"}


@app.delete("/activities/{activity_name}/unregister")
def unregister_from_activity(activity_name: str, email: str):
    """Unregister a student from an activity"""
    # Validate activity exists
    if activity_name not in activities:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Get the specific activity
    activity = activities[activity_name]

    # Validate student is signed up
    if email not in activity["participants"]:
        raise HTTPException(
            status_code=400,
            detail="Student is not signed up for this activity"
        )

    # Remove student
    activity["participants"].remove(email)
    return {"message": f"Unregistered {email} from {activity_name}"}


# ============================================================================
# Observability API Endpoints
# ============================================================================

@app.get("/api/observability")
def get_observability_data():
    """
    Get aggregated observability data for the dashboard.
    
    Returns summary statistics, recent timelines, metrics, and errors.
    """
    return _observability_data


@app.post("/api/observability")
def update_observability_data(data: dict):
    """
    Update observability data from an AgenticChain execution.
    
    This endpoint can be called by the agentic chain to push
    observability data to the dashboard.
    """
    global _observability_data
    
    # Update summary
    if "summary" in data:
        _observability_data["summary"].update(data["summary"])
    
    # Update LLM usage
    if "llm_usage" in data:
        _observability_data["llm_usage"].update(data["llm_usage"])
    
    # Add new timelines (keep last 50)
    if "recent_timelines" in data:
        _observability_data["recent_timelines"] = (
            data["recent_timelines"] + _observability_data["recent_timelines"]
        )[:50]
    
    # Update metrics
    if "metrics" in data:
        _observability_data["metrics"].update(data["metrics"])
    
    # Add new errors (keep last 100)
    if "recent_errors" in data:
        _observability_data["recent_errors"] = (
            data["recent_errors"] + _observability_data["recent_errors"]
        )[:100]
    
    # Update timestamp
    _observability_data["generated_at"] = datetime.now().isoformat()
    
    return {"status": "ok"}


@app.get("/api/metrics")
def get_metrics_prometheus():
    """
    Get metrics in Prometheus text format.
    
    This endpoint can be scraped by Prometheus for monitoring.
    """
    try:
        from agentic_chain.observability import MetricsCollector
        collector = MetricsCollector()
        
        # Add data from stored observability data
        summary = _observability_data["summary"]
        if summary["total_executions"] > 0:
            collector.set_gauge("executions_total", summary["total_executions"])
            collector.set_gauge("success_rate", summary["success_rate"])
        
        return collector.to_prometheus()
    except ImportError:
        return "# Metrics not available - agentic_chain not installed\n"


@app.get("/api/traces")
def get_traces(trace_id: Optional[str] = None):
    """
    Get trace data, optionally filtered by trace_id.
    """
    timelines = _observability_data["recent_timelines"]
    
    if trace_id:
        timelines = [t for t in timelines if t.get("trace_id") == trace_id]
    
    return {"traces": timelines}


@app.delete("/api/observability")
def clear_observability_data():
    """
    Clear all observability data.
    
    Useful for resetting the dashboard.
    """
    global _observability_data
    _observability_data = {
        "summary": {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "success_rate": 0.0,
            "avg_execution_time_ms": 0.0,
            "p95_execution_time_ms": 0.0,
        },
        "llm_usage": {
            "total_tokens": 0,
            "total_cost": 0.0,
        },
        "recent_timelines": [],
        "metrics": {},
        "recent_errors": [],
        "generated_at": None,
    }
    return {"status": "cleared"}
