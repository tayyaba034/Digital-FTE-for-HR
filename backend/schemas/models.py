"""
schemas/models.py
All Pydantic models for the Candidates FTE system.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class JobStatus(str, Enum):
    FETCHED = "fetched"
    SHORTLISTED = "shortlisted"
    EXCLUDED = "excluded"
    APPLIED = "applied"


class ApplicationStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SENT = "sent"
    REPLIED = "replied"
    INTERVIEW = "interview"
    REJECTED = "rejected"


class CVStatus(str, Enum):
    GENERATING = "generating"
    PENDING_APPROVAL = "pending_approval"
    EDITING = "editing"
    APPROVED = "approved"


class HITLType(str, Enum):
    CV_REVIEW = "cv_review"
    EMAIL_REVIEW = "email_review"


# ─────────────────────────────────────────────
# User Profile
# ─────────────────────────────────────────────

class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    target_roles: list[str] = []
    target_locations: list[str] = []
    skills: list[str] = []
    experience_years: Optional[int] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    raw_cv_text: Optional[str] = None
    cv_file_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────

class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    requirements: list[str] = []
    salary_range: Optional[str] = None
    job_type: Optional[str] = None  # full-time, contract, etc.
    source_platform: str  # linkedin, indeed, glassdoor
    source_url: str
    posted_date: Optional[datetime] = None
    match_score: float = 0.0
    status: JobStatus = JobStatus.FETCHED
    hr_email: Optional[str] = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class JobSearchRequest(BaseModel):
    query: str
    locations: list[str] = []
    max_results: int = 50


class JobSearchResponse(BaseModel):
    jobs: list[Job]
    total_fetched: int
    duplicates_removed: int
    search_query: str


# ─────────────────────────────────────────────
# CV / Resume
# ─────────────────────────────────────────────

class TailoredCV(BaseModel):
    id: str
    job_id: str
    job_title: str
    company: str
    content_markdown: str       # The full CV in markdown
    content_html: str           # Rendered HTML for preview
    ats_score: float = 0.0      # Estimated ATS compatibility
    keywords_matched: list[str] = []
    keywords_missing: list[str] = []
    status: CVStatus = CVStatus.PENDING_APPROVAL
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CVEditRequest(BaseModel):
    cv_id: str
    section: Optional[str] = None   # e.g., "summary", "experience"
    instruction: str                  # natural language edit instruction
    direct_content: Optional[str] = None  # if user edited directly in editor


class CVApprovalRequest(BaseModel):
    cv_ids: list[str]   # can approve multiple at once


# ─────────────────────────────────────────────
# Applications / Emails
# ─────────────────────────────────────────────

class EmailDraft(BaseModel):
    id: str
    job_id: str
    cv_id: str
    job_title: str
    company: str
    hr_email: str
    subject: str
    body: str
    cover_letter: str
    status: ApplicationStatus = ApplicationStatus.PENDING_APPROVAL
    sent_at: Optional[datetime] = None
    replied_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EmailEditRequest(BaseModel):
    email_id: str
    section: Optional[str] = None   # "subject", "body", "cover_letter"
    instruction: str
    direct_content: Optional[str] = None


class EmailApprovalRequest(BaseModel):
    email_ids: list[str]


# ─────────────────────────────────────────────
# HITL (Human-in-the-Loop) Events
# ─────────────────────────────────────────────

class HITLCheckpoint(BaseModel):
    id: str
    type: HITLType
    workflow_id: str
    payload: dict[str, Any]       # CVs or emails to review
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    approved: Optional[bool] = None


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class OrchestratorIntent(BaseModel):
    raw_query: str
    intent_type: str              # "full_pipeline", "job_search_only", etc.
    agents_to_activate: list[str]
    parameters: dict[str, Any] = {}


class WorkflowState(BaseModel):
    workflow_id: str
    status: AgentStatus = AgentStatus.IDLE
    current_agent: Optional[str] = None
    completed_agents: list[str] = []
    pending_hitl: Optional[str] = None   # HITL checkpoint id
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Chat / SSE Events
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str           # "user" | "agent" | "system"
    content: str
    agent_name: Optional[str] = None
    metadata: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentEvent(BaseModel):
    """Server-sent event pushed to frontend in real time."""
    event_type: str     # "agent_started" | "agent_progress" | "hitl_required" | "agent_done" | "error"
    agent_name: Optional[str] = None
    message: str
    data: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Interview Prep
# ─────────────────────────────────────────────

class InterviewQuestion(BaseModel):
    id: str
    job_id: str
    category: str       # "technical" | "behavioral" | "situational"
    question: str
    ideal_answer_hints: list[str] = []
    difficulty: str = "medium"


class MockInterviewSession(BaseModel):
    session_id: str
    job_id: str
    messages: list[ChatMessage] = []
    skill_gaps: list[str] = []
    resources: list[dict[str, str]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
