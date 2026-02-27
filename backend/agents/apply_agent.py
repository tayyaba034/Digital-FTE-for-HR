"""
agents/apply_agent.py
Drafts and sends job application emails via Gmail.
Full HITL: shows email drafts → user approves → agent sends.
"""
import uuid
from datetime import datetime
from typing import Any
import httpx
import os
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from memory.store import long_term, short_term
from observability.config import trace_agent
from schemas.models import AgentEvent, ApplicationStatus
from tools.gmail_sender import gmail_sender

log = structlog.get_logger()

EMAIL_DRAFT_SYSTEM = """You are a professional job application specialist who writes compelling, 
personalized application emails that get responses from HR professionals.

Rules:
1. Professional but warm tone — not robotic
2. Specific to the company and role — show you've done research
3. Lead with your strongest relevant qualification
4. Cover letter: 3-4 paragraphs max
5. Email body: brief intro (3-4 lines) that references the CV + cover letter attached
6. Never use generic phrases like "I am writing to express my interest"

Output ONLY valid JSON with this structure:
{
  "subject": "Application: Senior Python Engineer | [Your Name]",
  "email_body": "...",
  "cover_letter": "..."
}"""

EMAIL_EDIT_SYSTEM = """You are an expert email copywriter. Edit the following job application content 
based on the user's instruction.

Current content:
{current_content}

Return ONLY the edited content (same format — don't add explanation)."""

HR_EMAIL_LOOKUP_PROMPT = """Given a company name, try to find the likely HR/recruiting email.
Return ONLY a JSON object: {{"email": "hr@company.com", "confidence": 0.8}}
If unknown, return {{"email": null, "confidence": 0}}

Company: {company}"""


class ApplyAgent:
    def __init__(self, event_callback=None):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
        self.event_callback = event_callback
        self.hunter_api_key = os.getenv("HUNTER_IO_API_KEY")

    async def _emit(self, event_type: str, message: str, data: dict = {}):
        event = AgentEvent(event_type=event_type, agent_name="apply_agent", message=message, data=data)
        log.info("apply_agent.event", message=message)
        if self.event_callback:
            await self.event_callback(event)

    async def _find_hr_email(self, company: str, domain: str | None = None) -> str | None:
        """Look up HR email via Hunter.io or Claude inference."""
        # Try Hunter.io first
        if self.hunter_api_key and domain:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://api.hunter.io/v2/domain-search",
                        params={
                            "domain": domain,
                            "api_key": self.hunter_api_key,
                            "type": "personal",
                            "department": "hr",
                            "limit": 1,
                        }
                    )
                    data = resp.json()
                    emails = data.get("data", {}).get("emails", [])
                    if emails:
                        return emails[0].get("value")
            except Exception as e:
                log.error("apply_agent.hunter_error", error=str(e))

        # Fallback: Claude guesses common HR email patterns
        import json
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=HR_EMAIL_LOOKUP_PROMPT.format(company=company))
            ])
            result = json.loads(response.content)
            if result.get("confidence", 0) > 0.6:
                return result.get("email")
        except Exception:
            pass

        return None

    async def _draft_email(self, job: dict, cv: dict, user_profile: dict) -> dict:
        """Draft an application email + cover letter for a specific job."""
        import json
        
        profile_summary = {
            "name": user_profile.get("name", "Candidate"),
            "email": user_profile.get("email", ""),
            "skills": user_profile.get("data", {}).get("skills", []) if isinstance(user_profile.get("data"), dict) else [],
            "experience_years": user_profile.get("data", {}).get("experience_years"),
            "target_roles": user_profile.get("data", {}).get("target_roles", []) if isinstance(user_profile.get("data"), dict) else [],
        }

        messages = [
            SystemMessage(content=EMAIL_DRAFT_SYSTEM),
            HumanMessage(content=f"""
Candidate Profile:
{json.dumps(profile_summary, indent=2)}

Job Details:
- Title: {job['title']}
- Company: {job['company']}
- Location: {job.get('location', 'N/A')}
- Description excerpt: {job.get('description', '')[:800]}

CV Summary (key highlights):
{cv.get('content_markdown', '')[:1000]}

Write the application email and cover letter for this specific role.
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        try:
            return json.loads(response.content)
        except Exception:
            return {
                "subject": f"Application: {job['title']} | {user_profile.get('name', 'Candidate')}",
                "email_body": response.content[:500],
                "cover_letter": response.content[500:],
            }

    def _fallback_email(self, job: dict, user_profile: dict) -> dict:
        name = user_profile.get("name", "Candidate")
        subject = f"Application: {job.get('title', 'Role')} | {name}"
        body = (
            f"Dear Hiring Team,\n\n"
            f"I am applying for the {job.get('title', 'role')} position at {job.get('company', 'your company')}.\n"
            f"My profile and tailored CV are attached for review.\n\n"
            f"Best regards,\n{name}"
        )
        cover = (
            f"Dear Hiring Team,\n\n"
            f"I am excited to apply for the {job.get('title', 'role')} position at {job.get('company', 'your company')}.\n"
            f"I believe my background aligns with your requirements and I would welcome the opportunity to discuss further.\n\n"
            f"Sincerely,\n{name}"
        )
        return {"subject": subject, "email_body": body, "cover_letter": cover}

    async def edit_email(
        self,
        email_id: str,
        user_id: str,
        instruction: str,
        section: str | None = None,
        direct_content: str | None = None,
    ) -> dict:
        """Edit an email draft based on user instruction. HITL editing feature."""
        drafts = await long_term.get_email_drafts(user_id)
        draft = next((d for d in drafts if d["id"] == email_id), None)
        if not draft:
            return {"error": "Email draft not found"}

        if direct_content:
            # Direct edit — update the specific field
            field_map = {"subject": "subject", "body": "body", "cover_letter": "cover_letter"}
            field = field_map.get(section or "body", "body")
            draft[field] = direct_content
        else:
            # AI-assisted edit
            current = f"Subject: {draft.get('subject', '')}\n\nEmail Body:\n{draft.get('body', '')}\n\nCover Letter:\n{draft.get('cover_letter', '')}"
            messages = [
                SystemMessage(content=EMAIL_EDIT_SYSTEM.format(current_content=current)),
                HumanMessage(content=instruction)
            ]
            response = await self.llm.ainvoke(messages)
            
            # Try to parse if JSON, else apply to body
            import json
            try:
                updated = json.loads(response.content)
                draft.update(updated)
            except Exception:
                if section == "cover_letter":
                    draft["cover_letter"] = response.content
                else:
                    draft["body"] = response.content

        # Save updated draft
        await long_term.save_email_draft(draft, user_id)
        return draft

    async def approve_and_send(
        self,
        email_ids: list[str],
        user_id: str,
        gmail_creds: dict,
        cv_attachment: bytes | None = None,
    ) -> dict:
        """Send approved emails via Gmail."""
        sent = []
        failed = []

        drafts = await long_term.get_email_drafts(user_id)
        approved_drafts = [d for d in drafts if d["id"] in email_ids]

        for draft in approved_drafts:
            try:
                result = await gmail_sender.send_email(
                    creds_dict=gmail_creds,
                    to=draft["hr_email"],
                    subject=draft["subject"],
                    body=draft["body"],
                    cover_letter=draft["cover_letter"],
                    cv_attachment=cv_attachment,
                    cv_filename=f"CV_{draft.get('company', 'Application').replace(' ', '_')}.pdf",
                )

                if result.get("success"):
                    await long_term.update_email_status(draft["id"], ApplicationStatus.SENT, datetime.utcnow())
                    await long_term.update_job_status(draft["job_id"], "applied")
                    sent.append(draft["id"])
                    log.info("apply_agent.sent", company=draft.get("company"), email=draft["hr_email"])
                else:
                    failed.append({"id": draft["id"], "error": result.get("error")})

            except Exception as e:
                log.error("apply_agent.send_error", email_id=draft["id"], error=str(e))
                failed.append({"id": draft["id"], "error": str(e)})

        return {"sent": len(sent), "failed": len(failed), "sent_ids": sent, "failed_ids": failed}

    @trace_agent("apply_agent")
    async def run(self, user_id: str, workflow_id: str) -> dict[str, Any]:
        """
        Full apply pipeline:
        1. Load approved CVs
        2. Find HR emails
        3. Draft application emails
        4. Save drafts → trigger HITL for user approval
        """
        # Load approved CVs
        from schemas.models import CVStatus
        cvs = await long_term.get_cvs(user_id, status=CVStatus.APPROVED)
        if not cvs:
            await self._emit("agent_progress", "No approved CVs found. Please approve CVs in the CVs tab first.",
                            {"workflow_id": workflow_id})
            return {"total": 0, "drafts": []}

        user_profile = await long_term.get_user_profile(user_id) or {}
        jobs = await long_term.get_jobs(user_id)
        job_map = {j["id"]: j for j in jobs}

        draft_ids = []
        for i, cv in enumerate(cvs):
            job = job_map.get(cv["job_id"])
            if not job:
                continue

            await self._emit("agent_progress",
                           f"Drafting email {i+1}/{len(cvs)}: {job['title']} at {job['company']}...",
                           {"workflow_id": workflow_id})

            # Find HR email
            hr_email = job.get("hr_email")
            if not hr_email:
                hr_email = await self._find_hr_email(job["company"])
                if hr_email:
                    await long_term.update_job_status(job["id"], job.get("status", "shortlisted"), hr_email)

            if not hr_email:
                log.warning("apply_agent.no_hr_email", company=job["company"])
                hr_email = f"hr@{job['company'].lower().replace(' ', '')}.com"  # best guess fallback

            # Draft email
            try:
                try:
                    email_content = await self._draft_email(job, cv, user_profile)
                except Exception as e:
                    log.warning("apply_agent.draft_fallback", job_id=job["id"], error=str(e))
                    email_content = self._fallback_email(job, user_profile)
                
                draft = {
                    "id": str(uuid.uuid4()),
                    "job_id": job["id"],
                    "cv_id": cv["id"],
                    "job_title": job["title"],
                    "company": job["company"],
                    "hr_email": hr_email,
                    "subject": email_content.get("subject", f"Application: {job['title']}"),
                    "body": email_content.get("email_body", ""),
                    "cover_letter": email_content.get("cover_letter", ""),
                    "status": ApplicationStatus.PENDING_APPROVAL,
                }

                await long_term.save_email_draft(draft, user_id)
                draft_ids.append(draft["id"])

            except Exception as e:
                log.error("apply_agent.draft_error", job_id=job["id"], error=str(e))

        # Trigger HITL for email review
        checkpoint_id = str(uuid.uuid4())
        await short_term.set_hitl_checkpoint(checkpoint_id, {
            "type": "email_review",
            "workflow_id": workflow_id,
            "user_id": user_id,
            "draft_ids": draft_ids,
            "approved": None,
        })

        await self._emit("hitl_required",
                        f"{len(draft_ids)} email drafts ready. Please review and approve in the Applications tab.",
                        {"workflow_id": workflow_id, "checkpoint_id": checkpoint_id,
                         "type": "email_review", "draft_ids": draft_ids})

        auto_approve = os.getenv("AUTO_APPROVE_HITL", "false").lower() in ("1", "true", "yes")
        if auto_approve and draft_ids:
            for draft_id in draft_ids:
                await long_term.update_email_status(draft_id, ApplicationStatus.APPROVED)
            await short_term.resolve_hitl_checkpoint(checkpoint_id, True)
            await self._emit(
                "agent_progress",
                "Auto-approved email drafts.",
                {"workflow_id": workflow_id},
            )

            gmail_creds = await short_term.get(f"gmail_creds:{user_id}")
            if gmail_creds:
                send_result = await self.approve_and_send(draft_ids, user_id, gmail_creds)
                await self._emit(
                    "agent_progress",
                    f"Auto-send complete: {send_result.get('sent', 0)} sent, {send_result.get('failed', 0)} failed.",
                    {"workflow_id": workflow_id, "send_result": send_result},
                )

        return {"total": len(draft_ids), "draft_ids": draft_ids, "checkpoint_id": checkpoint_id}
