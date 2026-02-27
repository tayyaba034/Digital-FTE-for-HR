"""
api/main.py
FastAPI application — all REST endpoints + SSE streaming for real-time agent events.
"""
import asyncio
import json
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from agents.apply_agent import ApplyAgent
from agents.orchestrator import OrchestratorAgent
from agents.resume_agent import ResumeAgent
from memory.store import close_memory, init_memory, long_term, short_term
from observability.config import configure_observability
from schemas.models import (
    CVApprovalRequest,
    CVEditRequest,
    EmailApprovalRequest,
    EmailEditRequest,
    JobStatus,
)
from tools.cv_parser import parse_cv
from tools.gmail_sender import gmail_sender

log = structlog.get_logger()

# ─────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_observability()
    await init_memory()
    log.info("app.started")
    yield
    await close_memory()
    log.info("app.shutdown")


app = FastAPI(
    title="Candidates FTE API",
    description="Digital Full-Time Employee for autonomous job searching and applying",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000"), "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SSE Event Stream
# ─────────────────────────────────────────────

async def _event_generator(workflow_id: str) -> AsyncGenerator[str, None]:
    """Poll Redis for agent events and stream via SSE."""
    last_index = 0
    timeout_count = 0

    while timeout_count < 720:  # max 1 hour
        events = await short_term.get_list(f"events:{workflow_id}")
        
        if len(events) > last_index:
            for event in events[last_index:]:
                yield f"data: {json.dumps(event)}\n\n"
                
                if event.get("event_type") in ("agent_done", "error") and event.get("data", {}).get("completed"):
                    yield f"data: {json.dumps({'event_type': 'stream_end'})}\n\n"
                    return
                
                last_index = len(events)

        await asyncio.sleep(1)
        timeout_count += 1

    yield f"data: {json.dumps({'event_type': 'stream_timeout'})}\n\n"


# ─────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class UserProfileRequest(BaseModel):
    id: str
    name: str
    email: str
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    target_roles: list[str] = []
    target_locations: list[str] = []
    skills: list[str] = []
    experience_years: int | None = None
    salary_min: int | None = None
    salary_max: int | None = None


class MockInterviewMessage(BaseModel):
    session_id: str
    job_id: str
    message: str
    user_id: str


class EmailSendRequest(BaseModel):
    email_ids: list[str]
    user_id: str
    gmail_creds: dict

class SendApprovedRequest(BaseModel):
    email_ids: list[str] = []


class JobStatusUpdate(BaseModel):
    status: str


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "candidates-fte-api"}


# ─────────────────────────────────────────────
# User Profile
# ─────────────────────────────────────────────

@app.post("/profile")
async def create_or_update_profile(profile: UserProfileRequest):
    """Create or update user profile."""
    data = profile.model_dump()
    result = await long_term.upsert_user_profile(data)
    return {"success": True, "profile": result}


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    profile = await long_term.get_user_profile(user_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile


@app.post("/profile/{user_id}/upload-cv")
async def upload_cv(user_id: str, file: UploadFile = File(...)):
    """Upload and parse user's existing CV."""
    content = await file.read()
    parsed = parse_cv(content, file.filename or "cv.pdf")
    
    # Update profile with parsed CV data
    profile = await long_term.get_user_profile(user_id)
    if profile:
        profile_data = profile.get("data", {})
        if isinstance(profile_data, str):
            try:
                profile_data = json.loads(profile_data)
            except Exception:
                profile_data = {}
        profile_data["raw_cv_text"] = parsed["raw_text"]
        profile_data["cv_sections"] = parsed["sections"]
        profile_data["contact_info"] = parsed["contact_info"]
        profile_data["parsed_skills"] = parsed.get("skills", [])
        profile_data["parsed_projects"] = parsed.get("projects", [])

        await long_term.upsert_user_profile({
            **profile,
            "data": profile_data,
        })
    
    return {"success": True, "parsed": parsed}


# ─────────────────────────────────────────────
# Chat / Orchestrator
# ─────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Accepts natural language query, 
    runs orchestrator, returns workflow_id for SSE streaming.
    """
    # Store user message
    await short_term.add_message(req.session_id, {
        "role": "user",
        "content": req.message,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Run orchestrator in background
    workflow_id = str(uuid.uuid4())
    
    async def run_orchestrator():
        async def event_callback(event):
            await short_term.append_to_list(f"events:{workflow_id}", event.model_dump(mode="json"))
        
        orchestrator = OrchestratorAgent(event_callback=event_callback)
        await orchestrator.run(
            user_id=req.user_id,
            session_id=req.session_id,
            raw_query=req.message,
        )

    asyncio.create_task(run_orchestrator())
    
    return {"workflow_id": workflow_id, "status": "started"}


@app.get("/chat/stream/{workflow_id}")
async def stream_events(workflow_id: str):
    """SSE endpoint — streams real-time agent events to frontend."""
    return StreamingResponse(
        _event_generator(workflow_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    messages = await short_term.get_messages(session_id)
    return {"messages": messages}


@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    status = await short_term.get_workflow_status(workflow_id)
    return status or {"status": "not_found"}


@app.post("/workflow/{workflow_id}/interrupt")
async def interrupt_workflow(workflow_id: str):
    """Allow user to pause/interrupt a running workflow."""
    await short_term.set_workflow_status(workflow_id, {
        "status": "paused",
        "interrupted_at": datetime.utcnow().isoformat(),
    })
    return {"success": True, "message": "Workflow interrupted"}


# ─────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────

@app.get("/jobs/{user_id}")
async def get_jobs(user_id: str, status: str | None = None):
    jobs = await long_term.get_jobs(user_id, status=status)
    return {"jobs": jobs, "total": len(jobs)}


@app.patch("/jobs/{job_id}")
async def update_job_status(job_id: str, update: JobStatusUpdate):
    await long_term.update_job_status(job_id, update.status)
    return {"success": True}


@app.post("/jobs/{user_id}/shortlist")
async def shortlist_jobs(user_id: str, job_ids: list[str]):
    for job_id in job_ids:
        await long_term.update_job_status(job_id, JobStatus.SHORTLISTED)
    return {"success": True, "shortlisted": len(job_ids)}


# ─────────────────────────────────────────────
# CVs (HITL)
# ─────────────────────────────────────────────

@app.get("/cvs/{user_id}")
async def get_cvs(user_id: str, status: str | None = None):
    cvs = await long_term.get_cvs(user_id, status=status)
    return {"cvs": cvs, "total": len(cvs)}


@app.post("/cvs/edit")
async def edit_cv(req: CVEditRequest, user_id: str):
    """Edit a CV section via inline editor or chatbot instruction."""
    agent = ResumeAgent()
    result = await agent.edit_cv_section(
        cv_id=req.cv_id,
        user_id=user_id,
        instruction=req.instruction,
        section=req.section,
        direct_content=req.direct_content,
    )
    return result


@app.post("/cvs/approve")
async def approve_cvs(req: CVApprovalRequest, user_id: str):
    """Approve CVs — moves them to 'approved' status for Apply Agent."""
    from schemas.models import CVStatus
    for cv_id in req.cv_ids:
        await long_term.update_cv_status(cv_id, CVStatus.APPROVED)
    return {"success": True, "approved": len(req.cv_ids)}


@app.post("/hitl/{checkpoint_id}/resolve")
async def resolve_hitl(checkpoint_id: str, approved: bool):
    """Resolve a HITL checkpoint — approve or reject."""
    await short_term.resolve_hitl_checkpoint(checkpoint_id, approved)
    return {"success": True, "approved": approved}


# ─────────────────────────────────────────────
# Applications / Emails (HITL)
# ─────────────────────────────────────────────

@app.get("/applications/{user_id}")
async def get_applications(user_id: str, status: str | None = None):
    drafts = await long_term.get_email_drafts(user_id, status=status)
    return {"applications": drafts, "total": len(drafts)}


@app.post("/applications/edit")
async def edit_email(req: EmailEditRequest, user_id: str):
    """Edit an email draft via inline editor or chatbot instruction."""
    agent = ApplyAgent()
    result = await agent.edit_email(
        email_id=req.email_id,
        user_id=user_id,
        instruction=req.instruction,
        section=req.section,
        direct_content=req.direct_content,
    )
    return result


@app.post("/applications/send")
async def send_applications(req: EmailSendRequest):
    """Send approved application emails via Gmail."""
    agent = ApplyAgent()
    result = await agent.approve_and_send(
        email_ids=req.email_ids,
        user_id=req.user_id,
        gmail_creds=req.gmail_creds,
    )
    return result

@app.post("/applications/send-approved/{user_id}")
async def send_approved_applications(user_id: str, req: SendApprovedRequest | None = None):
    """Send approved application emails using stored Gmail OAuth credentials."""
    gmail_creds = await short_term.get(f"gmail_creds:{user_id}")
    if not gmail_creds:
        raise HTTPException(400, "Gmail is not connected for this user. Connect via /auth/gmail first.")

    approved = await long_term.get_email_drafts(user_id, status="approved")
    selected_ids = set((req.email_ids if req else []) or [])
    if selected_ids:
        approved = [d for d in approved if d.get("id") in selected_ids]

    if not approved:
        return {"sent": 0, "failed": 0, "sent_ids": [], "failed_ids": [], "message": "No approved drafts to send."}

    agent = ApplyAgent()
    result = await agent.approve_and_send(
        email_ids=[d["id"] for d in approved if d.get("id")],
        user_id=user_id,
        gmail_creds=gmail_creds,
    )
    return result


@app.post("/applications/approve")
async def approve_emails(req: EmailApprovalRequest, user_id: str):
    """Mark emails as approved (not sent yet — just approved for sending)."""
    from schemas.models import ApplicationStatus
    for email_id in req.email_ids:
        await long_term.update_email_status(email_id, ApplicationStatus.APPROVED)
    return {"success": True, "approved": len(req.email_ids)}


# ─────────────────────────────────────────────
# Gmail OAuth2
# ─────────────────────────────────────────────

@app.get("/auth/gmail")
async def gmail_auth(user_id: str):
    """Redirect user to Gmail OAuth2 authorization URL."""
    try:
        url = gmail_sender.get_auth_url(state=user_id)
    except Exception as e:
        raise HTTPException(500, f"Gmail OAuth configuration error: {str(e)}")
    return RedirectResponse(url=url)


@app.get("/auth/gmail/callback")
async def gmail_callback(code: str, user_id: str | None = None, state: str | None = None):
    """Exchange OAuth2 code for credentials and store in session."""
    resolved_user_id = user_id or state
    if not resolved_user_id:
        raise HTTPException(400, "Missing user_id. Start auth via /auth/gmail?user_id=<id>.")
    creds = gmail_sender.exchange_code(code)
    await short_term.set(f"gmail_creds:{resolved_user_id}", creds, ttl=86400 * 30)
    return {"success": True, "message": "Gmail connected successfully"}


@app.get("/auth/gmail/status/{user_id}")
async def gmail_status(user_id: str):
    creds = await short_term.get(f"gmail_creds:{user_id}")
    return {"connected": creds is not None}


# ─────────────────────────────────────────────
# Interview Prep
# ─────────────────────────────────────────────

@app.get("/interview/{user_id}/prep")
async def get_interview_prep(user_id: str):
    """Get all generated interview prep materials."""
    jobs = await long_term.get_jobs(user_id)
    preps = []
    for job in jobs:
        prep = await short_term.get(f"interview_prep:{job['id']}")
        if prep:
            preps.append(prep)
    return {"preps": preps}


@app.post("/interview/chat")
async def mock_interview_chat(req: MockInterviewMessage):
    """Handle a message in a mock interview session."""
    from agents.interview_agent import InterviewAgent
    
    jobs = await long_term.get_jobs(req.user_id)
    job = next((j for j in jobs if j["id"] == req.job_id), None)
    if not job:
        raise HTTPException(404, "Job not found")

    agent = InterviewAgent()
    response = await agent.mock_interview_chat(
        session_id=req.session_id,
        user_message=req.message,
        job=job,
    )
    return {"response": response}


# ─────────────────────────────────────────────
# Observability
# ─────────────────────────────────────────────

@app.get("/observability/{workflow_id}/traces")
async def get_traces(workflow_id: str):
    traces = await long_term.get_traces(workflow_id)
    return {"traces": traces}


@app.get("/observability/{user_id}/events")
async def get_recent_events(user_id: str):
    """Get recent agent events for the user."""
    # Return last 50 events from all workflows
    events = await short_term.get_list(f"user_events:{user_id}")
    return {"events": events[-50:]}
