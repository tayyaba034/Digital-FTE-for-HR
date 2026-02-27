"""
chainlit_app.py — Candidates FTE
A delegation-first interface: you give intent, the system executes end-to-end.
Built with Chainlit + FastAPI backend.
"""
import asyncio
import json
import os
import sys
import uuid
import httpx

import chainlit as cl
from chainlit.element import Text, File, Pdf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
BACKEND_PATH = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, BACKEND_PATH)

# Try to import agents directly (faster than HTTP for same process)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BACKEND_PATH, ".env"))
    from memory.store import init_memory, short_term, long_term
    from agents.orchestrator import OrchestratorAgent
    from agents.resume_agent import ResumeAgent
    from agents.apply_agent import ApplyAgent
    from tools.cv_parser import parse_cv
    DIRECT_MODE = True
except Exception as e:
    print(f"Running in HTTP mode: {e}")
    DIRECT_MODE = False


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _user_id():
    return cl.user_session.get("user_id", "anon-user")

def _session_id():
    return cl.user_session.get("session_id", f"sess-{uuid.uuid4()}")


async def _api(method: str, path: str, **kwargs):
    """Call the FastAPI backend."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await getattr(client, method)(f"{API_BASE}{path}", **kwargs)
        r.raise_for_status()
        return r.json()


def _format_agent_name(name: str) -> str:
    return {
        "job_search_agent":  "Job Search",
        "resume_agent":      "Resume Builder",
        "apply_agent":       "Application Agent",
        "interview_agent":   "Interview Coach",
        "orchestrator":      "Orchestrator",
    }.get(name, name.replace("_", " ").title())


# ──────────────────────────────────────────────────────────────
# App Lifecycle
# ──────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    """Initialize session and onboard user."""
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("profile", {})
    cl.user_session.set("workflow_id", None)

    if DIRECT_MODE:
        await init_memory()

    welcome = cl.Message(
        content=(
            "**Welcome to Candidates FTE — your delegated job search and application engine.**\n\n"
            "I work as your digital coordinator:\n"
            "- Search job boards (LinkedIn, Indeed, Glassdoor)\n"
            "- Tailor your CV for each shortlisted role\n"
            "- Draft and send applications via Gmail\n"
            "- Prepare you for interviews\n\n"
            "**To get started, I need two things from you:**\n\n"
            "1. Tell me briefly about yourself — your name, current role, skills, and what you're looking for\n"
            "2. Upload your current CV (PDF or DOCX) using the paperclip icon below\n\n"
            "*Or, if you're already set up, just tell me what you'd like done today.*"
        ),
        author="Candidates FTE",
    )
    await welcome.send()


# ──────────────────────────────────────────────────────────────
# File Upload Handling
# ──────────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    """Handle all user messages (text and file uploads)."""
    user_id = _user_id()
    session_id = _session_id()

    # ── Check for file uploads ─────────────────
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, 'path') and elem.path:
                await _handle_cv_upload(elem, user_id, message.content)
                return

    # ── Route text message ─────────────────────
    content = message.content.strip()
    if not content:
        return

    profile = cl.user_session.get("profile", {})

    # Check if user is providing profile info (early stage onboarding)
    if not profile.get("name") and not any(
        keyword in content.lower()
        for keyword in ["find", "search", "apply", "job", "interview", "cv", "resume", "tailor"]
    ):
        await _handle_profile_setup(content, user_id)
        return

    # Otherwise, treat as a delegation request
    await _handle_delegation(content, user_id, session_id)


async def _handle_cv_upload(elem, user_id: str, user_message: str):
    """Parse uploaded CV and store it in the user's profile."""
    msg = cl.Message(content="Parsing your CV...", author="Candidates FTE")
    await msg.send()

    try:
        with open(elem.path, "rb") as f:
            file_bytes = f.read()

        filename = getattr(elem, 'name', 'cv.pdf')

        if DIRECT_MODE:
            parsed = parse_cv(file_bytes, filename)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{API_BASE}/profile/{user_id}/upload-cv",
                    files={"file": (filename, file_bytes)},
                )
                parsed = r.json().get("parsed", {})

        # Store in session
        profile = cl.user_session.get("profile", {})
        profile["raw_cv_text"] = parsed.get("raw_text", "")
        profile["cv_sections"] = parsed.get("sections", {})
        profile["contact_info"] = parsed.get("contact_info", {})
        profile["cv_filename"] = filename
        cl.user_session.set("profile", profile)

        # Extract highlights for confirmation
        contact = parsed.get("contact_info", {})
        sections = parsed.get("sections", {})

        summary_lines = [f"**CV parsed successfully** — {parsed.get('word_count', 0)} words\n"]
        if contact.get("email"):
            summary_lines.append(f"- Email: {contact['email']}")
        if contact.get("linkedin"):
            summary_lines.append(f"- LinkedIn: {contact['linkedin']}")
        if sections.get("skills"):
            preview = sections["skills"][:200].replace("\n", ", ")
            summary_lines.append(f"- Skills detected: {preview}...")
        if sections.get("experience"):
            summary_lines.append(f"- Experience section found")

        summary_lines.append(
            "\nYour CV is on file. Now tell me what you'd like to do:\n"
            "- *\"Find senior Python engineer roles in Pakistan and apply to the top 5\"*\n"
            "- *\"Search for remote ML engineer jobs\"*\n"
            "- *\"Prepare me for Google interviews\"*"
        )

        msg.content = "\n".join(summary_lines)
        await msg.update()

    except Exception as e:
        msg.content = f"Could not parse the CV: {str(e)}. Please try a PDF or DOCX file."
        await msg.update()


async def _handle_profile_setup(content: str, user_id: str):
    """Extract profile info from natural language and confirm."""
    profile = cl.user_session.get("profile", {})

    # Simple keyword extraction for common fields
    lines = content.lower()
    if not profile.get("name"):
        # Try to find a name (first capitalized words)
        import re
        name_match = re.search(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", content)
        if name_match:
            profile["name"] = name_match.group(1)

    profile["bio_text"] = content
    cl.user_session.set("profile", profile)

    # Store profile in backend
    if DIRECT_MODE:
        await long_term.upsert_user_profile({
            "id": _user_id(),
            "name": profile.get("name", "Candidate"),
            "email": profile.get("contact_info", {}).get("email", ""),
            "data": profile,
        })

    msg = cl.Message(
        content=(
            f"Got it. I've noted your background.\n\n"
            "If you haven't already, please **upload your CV** using the paperclip icon — "
            "this lets me tailor applications precisely to each role.\n\n"
            "Or if you're ready, just tell me what you'd like done:\n"
            "- *\"Find Python engineer roles in Pakistan and apply to the top 5\"*\n"
            "- *\"Search for remote data scientist jobs\"*"
        ),
        author="Candidates FTE",
    )
    await msg.send()


# ──────────────────────────────────────────────────────────────
# Main Delegation Handler
# ──────────────────────────────────────────────────────────────

async def _handle_delegation(content: str, user_id: str, session_id: str):
    """Core delegation — run orchestrator, stream progress, handle HITL."""
    workflow_id = str(uuid.uuid4())
    cl.user_session.set("workflow_id", workflow_id)

    # Show plan message
    planning_msg = cl.Message(
        content="Analysing your request and building an execution plan...",
        author="Candidates FTE",
    )
    await planning_msg.send()

    # Task list for real-time agent progress
    task_list = cl.TaskList(status="Running")
    task_parse = cl.Task(title="Parsing intent", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task_parse)
    await task_list.send()

    if DIRECT_MODE:
        await _run_agents_direct(
            content, user_id, session_id, workflow_id,
            planning_msg, task_list, task_parse
        )
    else:
        await _run_agents_http(
            content, user_id, session_id, workflow_id,
            planning_msg, task_list, task_parse
        )


async def _run_agents_direct(
    content, user_id, session_id, workflow_id,
    planning_msg, task_list, task_parse
):
    """Run agents in-process with live streaming via Chainlit."""
    received_events = []
    agent_tasks: dict[str, cl.Task] = {}

    async def event_callback(event):
        received_events.append(event)
        agent = event.agent_name
        msg_text = event.message
        etype = event.event_type

        # Update or create task for this agent
        friendly = _format_agent_name(agent)
        if agent not in agent_tasks:
            task = cl.Task(title=friendly, status=cl.TaskStatus.RUNNING)
            await task_list.add_task(task)
            agent_tasks[agent] = task
        else:
            task = agent_tasks[agent]

        if etype == "agent_done":
            task.status = cl.TaskStatus.DONE
        elif etype == "error":
            task.status = cl.TaskStatus.FAILED

        task.forename = msg_text[:80]
        await task_list.send()

        # Also stream event as chat status
        if etype == "hitl_required":
            await _show_hitl_prompt(event, user_id, workflow_id)

    try:
        # Ensure memory is initialized
        await init_memory()

        # Seed user profile if we have CV data
        profile = cl.user_session.get("profile", {})
        if profile:
            await long_term.upsert_user_profile({
                "id": user_id,
                "name": profile.get("name", "Candidate"),
                "email": profile.get("contact_info", {}).get("email", "candidate@email.com"),
                "data": profile,
            })

        task_parse.status = cl.TaskStatus.DONE
        task_parse.forename = "Intent parsed"
        await task_list.send()

        orchestrator = OrchestratorAgent(event_callback=event_callback)
        result = await orchestrator.run(
            user_id=user_id,
            session_id=session_id,
            raw_query=content,
        )

        # Show completion summary
        await _show_completion(result, user_id, workflow_id, task_list)

    except Exception as e:
        task_parse.status = cl.TaskStatus.FAILED
        await task_list.send()
        await cl.Message(
            content=f"An error occurred: {str(e)}\n\nPlease try again or rephrase your request.",
            author="Candidates FTE",
        ).send()


async def _run_agents_http(
    content, user_id, session_id, workflow_id,
    planning_msg, task_list, task_parse
):
    """Run agents via HTTP API with SSE streaming."""
    try:
        resp = await _api("post", "/chat", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": content,
        })
        wid = resp.get("workflow_id", workflow_id)

        task_parse.status = cl.TaskStatus.DONE
        task_parse.forename = "Intent received"
        await task_list.send()

        # Stream SSE events
        agent_tasks = {}
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("GET", f"{API_BASE}/chat/stream/{wid}") as stream:
                async for line in stream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        agent = data.get("agent_name", "orchestrator")
                        friendly = _format_agent_name(agent)
                        etype = data.get("event_type", "")
                        msg_text = data.get("message", "")

                        if agent not in agent_tasks:
                            task = cl.Task(title=friendly, status=cl.TaskStatus.RUNNING)
                            await task_list.add_task(task)
                            agent_tasks[agent] = task
                        else:
                            task = agent_tasks[agent]

                        task.forename = msg_text[:80]
                        if etype == "agent_done":
                            task.status = cl.TaskStatus.DONE
                        await task_list.send()

                        if etype in ("stream_end", "stream_timeout"):
                            break

                        if etype == "hitl_required":
                            # Show HITL modal
                            class FakeEvent:
                                def __init__(self, d):
                                    self.data = d
                                    self.message = d.get("message", "")
                            await _show_hitl_prompt(FakeEvent(data.get("data", {})), user_id, wid)

                    except json.JSONDecodeError:
                        continue

        await _show_completion({}, user_id, wid, task_list)

    except Exception as e:
        await cl.Message(
            content=f"Connection error: {str(e)}",
            author="Candidates FTE",
        ).send()


# ──────────────────────────────────────────────────────────────
# HITL: CV Review
# ──────────────────────────────────────────────────────────────

async def _show_hitl_prompt(event, user_id: str, workflow_id: str):
    """Show the appropriate HITL UI based on checkpoint type."""
    data = getattr(event, "data", {}) or {}
    hitl_type = data.get("type", "")

    if hitl_type == "cv_review":
        await _show_cv_review(user_id, data.get("checkpoint_id", ""), workflow_id)
    elif hitl_type == "email_review":
        await _show_email_review(user_id, data.get("checkpoint_id", ""), data.get("draft_ids", []), workflow_id)
    else:
        # Generic HITL
        actions = [
            cl.Action(name="approve_generic", value="approve", label="Approve and Continue"),
            cl.Action(name="reject_generic", value="reject", label="Stop"),
        ]
        await cl.Message(
            content=f"**Action Required:** {event.message}\n\nPlease review and approve to continue.",
            actions=actions,
            author="Candidates FTE",
        ).send()


async def _show_cv_review(user_id: str, checkpoint_id: str, workflow_id: str):
    """Display all tailored CVs for review — one card per CV with edit + approve."""
    if DIRECT_MODE:
        cvs = await long_term.get_cvs(user_id)
    else:
        data = await _api("get", f"/cvs/{user_id}")
        cvs = data.get("cvs", [])

    if not cvs:
        await cl.Message(
            content="No tailored CVs found. The resume agent may still be running.",
            author="Candidates FTE",
        ).send()
        return

    intro = cl.Message(
        content=(
            f"**{len(cvs)} tailored CV(s) are ready for your review.**\n\n"
            "Each CV has been customised for the specific role. "
            "You can **approve** each one, or use the chat below to request edits "
            "(e.g. *\"Make the summary more concise for the Google CV\"*).\n\n"
            "Once all CVs are approved, applications will be drafted and sent for your final approval."
        ),
        author="Candidates FTE",
    )
    await intro.send()

    cl.user_session.set("pending_cvs", {cv["id"]: False for cv in cvs})
    cl.user_session.set("cv_data", {cv["id"]: cv for cv in cvs})

    for cv in cvs:
        ats_pct = int(cv.get("ats_score", 0.7) * 100)
        keywords_matched = cv.get("keywords_matched", [])
        keywords_missing = cv.get("keywords_missing", [])

        # CV content as a collapsible text element
        cv_content = cv.get("content_markdown", "No content available")
        cv_element = cl.Text(
            name=f"cv_{cv['id'][:8]}",
            content=cv_content,
            display="side",
        )

        actions = [
            cl.Action(
                name="approve_cv",
                value=cv["id"],
                label=f"Approve for {cv.get('company', 'this role')}",
            ),
            cl.Action(
                name="request_cv_edit",
                value=cv["id"],
                label="Request Edit",
            ),
        ]

        card_text = (
            f"**{cv.get('job_title', 'Role')} — {cv.get('company', 'Company')}**\n\n"
            f"ATS Score: **{ats_pct}%**\n"
        )
        if keywords_matched:
            card_text += f"Keywords matched: {', '.join(keywords_matched[:6])}\n"
        if keywords_missing:
            card_text += f"Keywords to add: {', '.join(keywords_missing[:4])}\n"
        card_text += "\n*Click 'View CV' in the side panel to read and copy the full content.*"

        await cl.Message(
            content=card_text,
            elements=[cv_element],
            actions=actions,
            author="Candidates FTE",
        ).send()

    cl.user_session.set("current_hitl_type", "cv_review")
    cl.user_session.set("current_checkpoint_id", checkpoint_id)


@cl.action_callback("approve_cv")
async def approve_cv(action: cl.Action):
    """User approved one CV."""
    cv_id = action.value
    pending = cl.user_session.get("pending_cvs", {})
    pending[cv_id] = True
    cl.user_session.set("pending_cvs", pending)

    if DIRECT_MODE:
        from schemas.models import CVStatus
        await long_term.update_cv_status(cv_id, CVStatus.APPROVED)
    else:
        await _api("post", "/cvs/approve", json={"cv_ids": [cv_id]}, params={"user_id": _user_id()})

    await cl.Message(
        content=f"CV approved.",
        author="Candidates FTE",
    ).send()

    # Check if all CVs are approved
    if all(pending.values()):
        checkpoint_id = cl.user_session.get("current_checkpoint_id", "")
        await cl.Message(
            content=(
                "**All CVs approved.** Moving to the application drafting stage.\n\n"
                "I will now find HR contact details and draft personalised application emails "
                "with cover letters for each role. You will review all emails before anything is sent."
            ),
            author="Candidates FTE",
        ).send()

        if DIRECT_MODE:
            await short_term.resolve_hitl_checkpoint(checkpoint_id, True)
        else:
            await _api("post", f"/hitl/{checkpoint_id}/resolve", params={"approved": "true"})


@cl.action_callback("request_cv_edit")
async def request_cv_edit(action: cl.Action):
    """User wants to edit a specific CV via chat."""
    cv_id = action.value
    cv_data = cl.user_session.get("cv_data", {}).get(cv_id, {})
    cl.user_session.set("editing_cv_id", cv_id)

    await cl.Message(
        content=(
            f"**Editing CV for {cv_data.get('job_title', 'this role')} at {cv_data.get('company', 'company')}**\n\n"
            "Describe what you'd like to change, for example:\n"
            "- *\"Make the summary 2 sentences max\"*\n"
            "- *\"Add more emphasis on Python and FastAPI\"*\n"
            "- *\"Rewrite the experience bullet points with stronger action verbs\"*\n\n"
            "I will apply your edit and show you the updated CV."
        ),
        author="Candidates FTE",
    ).send()
    cl.user_session.set("current_hitl_type", "cv_edit")


# ──────────────────────────────────────────────────────────────
# HITL: Email Review
# ──────────────────────────────────────────────────────────────

async def _show_email_review(user_id: str, checkpoint_id: str, draft_ids: list, workflow_id: str):
    """Display all email drafts for review."""
    if DIRECT_MODE:
        drafts = await long_term.get_email_drafts(user_id)
    else:
        data = await _api("get", f"/applications/{user_id}")
        drafts = data.get("applications", [])

    if not drafts:
        await cl.Message(
            content="No email drafts found yet.",
            author="Candidates FTE",
        ).send()
        return

    await cl.Message(
        content=(
            f"**{len(drafts)} application email(s) are ready for your approval.**\n\n"
            "Review each email carefully — once approved, they will be sent via your Gmail account. "
            "You can request edits before approving.\n\n"
            "**Nothing has been sent yet.**"
        ),
        author="Candidates FTE",
    ).send()

    cl.user_session.set("pending_emails", {d["id"]: False for d in drafts})
    cl.user_session.set("email_data", {d["id"]: d for d in drafts})

    for draft in drafts:
        subject = draft.get("subject", "(No subject)")
        hr_email = draft.get("hr_email", "Unknown")
        body = draft.get("body", "")
        cover = draft.get("cover_letter", "")

        full_email = f"**To:** {hr_email}\n**Subject:** {subject}\n\n---\n\n{body}"
        if cover:
            full_email += f"\n\n---\n**Cover Letter:**\n\n{cover}"

        email_element = cl.Text(
            name=f"email_{draft['id'][:8]}",
            content=full_email,
            display="side",
        )

        actions = [
            cl.Action(
                name="approve_email",
                value=draft["id"],
                label=f"Approve — {draft.get('company', 'company')}",
            ),
            cl.Action(
                name="request_email_edit",
                value=draft["id"],
                label="Request Edit",
            ),
        ]

        await cl.Message(
            content=(
                f"**{draft.get('job_title', 'Role')} — {draft.get('company', 'Company')}**\n"
                f"To: `{hr_email}`\n"
                f"Subject: *{subject}*\n\n"
                "*Click 'View email' to read the full email and cover letter.*"
            ),
            elements=[email_element],
            actions=actions,
            author="Candidates FTE",
        ).send()

    cl.user_session.set("current_hitl_type", "email_review")
    cl.user_session.set("current_checkpoint_id", checkpoint_id)


@cl.action_callback("approve_email")
async def approve_email(action: cl.Action):
    """User approved one email draft."""
    email_id = action.value
    pending = cl.user_session.get("pending_emails", {})
    pending[email_id] = True
    cl.user_session.set("pending_emails", pending)

    if DIRECT_MODE:
        from schemas.models import ApplicationStatus
        await long_term.update_email_status(email_id, ApplicationStatus.APPROVED)
    else:
        await _api("post", "/applications/approve", json={"email_ids": [email_id]}, params={"user_id": _user_id()})

    await cl.Message(content="Email approved.", author="Candidates FTE").send()

    if all(pending.values()):
        checkpoint_id = cl.user_session.get("current_checkpoint_id", "")

        await cl.Message(
            content=(
                "**All emails approved.** Ready to send.\n\n"
                "To send via Gmail, I need your Gmail account to be connected. "
                f"Please visit: [Connect Gmail](http://localhost:8000/auth/gmail?user_id={_user_id()})\n\n"
                "Once connected, your applications will be dispatched automatically. "
                "I will track replies and update the status here."
            ),
            author="Candidates FTE",
        ).send()

        if DIRECT_MODE:
            await short_term.resolve_hitl_checkpoint(checkpoint_id, True)


@cl.action_callback("request_email_edit")
async def request_email_edit(action: cl.Action):
    """User wants to edit a specific email."""
    email_id = action.value
    email_data = cl.user_session.get("email_data", {}).get(email_id, {})
    cl.user_session.set("editing_email_id", email_id)
    cl.user_session.set("current_hitl_type", "email_edit")

    await cl.Message(
        content=(
            f"**Editing email for {email_data.get('company', 'this company')}**\n\n"
            "Tell me what to change:\n"
            "- *\"Make the opening line more direct\"*\n"
            "- *\"Shorten the cover letter to 3 paragraphs\"*\n"
            "- *\"Add a reference to my open-source work\"*"
        ),
        author="Candidates FTE",
    ).send()


# ──────────────────────────────────────────────────────────────
# Inline Editing via Chat
# ──────────────────────────────────────────────────────────────

async def _handle_cv_edit_request(content: str, user_id: str):
    """Apply a chat-based CV edit request."""
    cv_id = cl.user_session.get("editing_cv_id")
    if not cv_id:
        return False

    editing_msg = cl.Message(content="Applying your edit...", author="Candidates FTE")
    await editing_msg.send()

    try:
        if DIRECT_MODE:
            agent = ResumeAgent()
            result = await agent.edit_cv_section(
                cv_id=cv_id,
                user_id=user_id,
                instruction=content,
            )
        else:
            result = await _api(
                "post", "/cvs/edit",
                json={"cv_id": cv_id, "instruction": content},
                params={"user_id": user_id},
            )

        new_markdown = result.get("content_markdown", "")
        cv_element = cl.Text(
            name=f"cv_edited_{cv_id[:8]}",
            content=new_markdown,
            display="side",
        )

        cv_data = cl.user_session.get("cv_data", {})
        if cv_id in cv_data:
            cv_data[cv_id]["content_markdown"] = new_markdown
            cl.user_session.set("cv_data", cv_data)

        actions = [
            cl.Action(name="approve_cv", value=cv_id, label="Approve this version"),
            cl.Action(name="request_cv_edit", value=cv_id, label="Make another change"),
        ]

        editing_msg.content = "Edit applied. Here is the updated CV:"
        await editing_msg.update()

        await cl.Message(
            content="Review the updated CV in the side panel.",
            elements=[cv_element],
            actions=actions,
            author="Candidates FTE",
        ).send()
        return True

    except Exception as e:
        editing_msg.content = f"Could not apply the edit: {str(e)}"
        await editing_msg.update()
        return True


async def _handle_email_edit_request(content: str, user_id: str):
    """Apply a chat-based email edit request."""
    email_id = cl.user_session.get("editing_email_id")
    if not email_id:
        return False

    editing_msg = cl.Message(content="Revising the email...", author="Candidates FTE")
    await editing_msg.send()

    try:
        if DIRECT_MODE:
            agent = ApplyAgent()
            result = await agent.edit_email(
                email_id=email_id,
                user_id=user_id,
                instruction=content,
            )
        else:
            result = await _api(
                "post", "/applications/edit",
                json={"email_id": email_id, "instruction": content},
                params={"user_id": user_id},
            )

        new_body = result.get("body", "")
        new_cover = result.get("cover_letter", "")
        full = f"**Subject:** {result.get('subject', '')}\n\n{new_body}"
        if new_cover:
            full += f"\n\n---\n**Cover Letter:**\n\n{new_cover}"

        email_element = cl.Text(
            name=f"email_edited_{email_id[:8]}",
            content=full,
            display="side",
        )

        email_data = cl.user_session.get("email_data", {})
        if email_id in email_data:
            email_data[email_id].update({"body": new_body, "cover_letter": new_cover})
            cl.user_session.set("email_data", email_data)

        actions = [
            cl.Action(name="approve_email", value=email_id, label="Approve and send"),
            cl.Action(name="request_email_edit", value=email_id, label="Revise further"),
        ]

        editing_msg.content = "Email revised. Review the updated version:"
        await editing_msg.update()

        await cl.Message(
            content="See the side panel for the full updated email.",
            elements=[email_element],
            actions=actions,
            author="Candidates FTE",
        ).send()
        return True

    except Exception as e:
        editing_msg.content = f"Could not revise the email: {str(e)}"
        await editing_msg.update()
        return True


# ──────────────────────────────────────────────────────────────
# Message Router (continued to handle edits mid-HITL)
# ──────────────────────────────────────────────────────────────
# We patch on_message to also check HITL state
_original_on_message = on_message.__wrapped__ if hasattr(on_message, '__wrapped__') else None


@cl.on_message
async def on_message(message: cl.Message):  # noqa: F811
    """Handle all messages — delegates, profile setup, and mid-HITL edits."""
    user_id = _user_id()
    session_id = _session_id()

    # File upload check
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, 'path') and elem.path:
                await _handle_cv_upload(elem, user_id, message.content)
                return

    content = message.content.strip()
    if not content:
        return

    hitl_type = cl.user_session.get("current_hitl_type", "")

    # Route to appropriate edit handler if mid-HITL
    if hitl_type == "cv_edit":
        handled = await _handle_cv_edit_request(content, user_id)
        if handled:
            cl.user_session.set("current_hitl_type", "cv_review")
            return

    if hitl_type == "email_edit":
        handled = await _handle_email_edit_request(content, user_id)
        if handled:
            cl.user_session.set("current_hitl_type", "email_review")
            return

    # Profile setup check
    profile = cl.user_session.get("profile", {})
    if not profile.get("name") and not profile.get("bio_text") and not any(
        keyword in content.lower()
        for keyword in ["find", "search", "apply", "job", "interview", "cv", "resume", "tailor",
                        "prepare", "roles", "engineer", "developer", "manager", "analyst"]
    ):
        await _handle_profile_setup(content, user_id)
        return

    # Standard delegation
    await _handle_delegation(content, user_id, session_id)


# ──────────────────────────────────────────────────────────────
# Completion Summary
# ──────────────────────────────────────────────────────────────

async def _show_completion(result: dict, user_id: str, workflow_id: str, task_list: cl.TaskList):
    """Show a final summary after the workflow completes."""
    task_list.status = "Done"
    await task_list.send()

    await cl.Message(
        content=(
            "**Workflow complete.**\n\n"
            "Here is what was accomplished:\n"
            "- Job search and deduplication: done\n"
            "- CV tailoring per role: done\n"
            "- Application drafts: ready for your review\n\n"
            "If CVs or emails appeared above, please review and approve them. "
            "Otherwise you can give me a new instruction, for example:\n"
            "- *\"Show me my interview prep for the Stripe role\"*\n"
            "- *\"Search for more data scientist roles in Berlin\"*"
        ),
        author="Candidates FTE",
    ).send()
