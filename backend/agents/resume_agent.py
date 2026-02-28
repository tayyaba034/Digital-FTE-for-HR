"""
agents/resume_agent.py
Tailors the user's CV for each shortlisted job.
ATS-optimized, keyword-aligned, section-by-section generation.
Inspired by BowJob prompts (https://github.com/rurahim/BowJob)
"""
import uuid
from datetime import datetime
from typing import Any
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from memory.store import long_term, short_term
from observability.config import trace_agent
from schemas.models import AgentEvent, CVStatus
from tools.cv_parser import parse_cv_sections

log = structlog.get_logger()

# ─────────────────────────────────────────────
# Prompts (BowJob-inspired)
# ─────────────────────────────────────────────

CV_TAILORING_SYSTEM = """You are an expert CV writer and ATS optimization specialist.
Your task is to tailor a candidate's existing CV to perfectly match a specific job description.

Rules:
1. Keep the candidate's real experience — never fabricate or exaggerate
2. Reorder and emphasize relevant experience for this specific role
3. Mirror keywords from the job description naturally throughout the CV
4. Use strong action verbs: Built, Designed, Led, Deployed, Reduced, Increased...
5. Quantify achievements wherever possible (%, £, time saved)
6. Keep it to 1-2 pages
7. Write in clean Markdown format with clear sections

Output the CV in Markdown with these sections (skip if not applicable):
# [Full Name]
**Contact** | email | phone | linkedin | github

## Professional Summary
(2-3 sentences tailored to this specific role)

## Key Skills
(bullet list of most relevant skills for THIS job)

## Professional Experience
(most recent first, emphasize relevant responsibilities)

## Education

## Certifications (if any)
"""

CV_SECTION_EDIT_SYSTEM = """You are an expert CV editor. The user wants to modify a specific section of their tailored CV.

Original CV section:
{section_content}

Job context:
{job_context}

Apply the user's edit instruction and return ONLY the updated section content in Markdown.
Do not include any explanation or preamble."""

ATS_SCORE_PROMPT = """Analyze this CV against the job description and return a JSON object:
{{
  "ats_score": 0.85,  // 0-1 score of ATS compatibility
  "keywords_matched": ["Python", "FastAPI", "AWS"],
  "keywords_missing": ["Kubernetes", "Terraform"],
  "improvements": ["Add more quantified achievements", "Include Kubernetes in skills"]
}}

Job Description:
{jd}

CV:
{cv}

Return ONLY valid JSON."""


def _cv_markdown_to_html(markdown: str) -> str:
    """Simple markdown to HTML conversion for preview."""
    import re
    html = markdown
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    # Bullet points
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', html)
    # Line breaks
    html = html.replace('\n\n', '</p><p>').replace('\n', '<br/>')
    html = f'<div class="cv-content"><p>{html}</p></div>'
    return html


def _job_keyword_set(job: dict) -> set[str]:
    text = " ".join([
        str(job.get("title", "")),
        str(job.get("description", "")),
        " ".join(job.get("requirements", []) or []),
    ]).lower()
    tokens = set()
    for tok in text.replace("/", " ").replace("-", " ").split():
        t = tok.strip(".,:;()[]{}")
        if len(t) >= 3:
            tokens.add(t)
    return tokens


def _project_score(project: dict, job_keywords: set[str]) -> int:
    blob = f"{project.get('title', '')} {project.get('summary', '')}".lower()
    score = 0
    for kw in job_keywords:
        if kw in blob:
            score += 1
    for sk in project.get("keywords", []) or []:
        if sk.lower() in job_keywords:
            score += 2
    return score


def _select_relevant_projects(parsed_projects: list[dict], job: dict, limit: int = 4) -> list[dict]:
    if not parsed_projects:
        return []
    job_keywords = _job_keyword_set(job)
    ranked = sorted(parsed_projects, key=lambda p: _project_score(p, job_keywords), reverse=True)
    selected = [p for p in ranked if _project_score(p, job_keywords) > 0]
    if not selected:
        selected = ranked
    return selected[:limit]


def _match_skills_to_job(skills: list[str], job: dict, limit: int = 16) -> tuple[list[str], list[str]]:
    if not skills:
        return [], []
    job_keywords = _job_keyword_set(job)
    matched: list[str] = []
    unmatched: list[str] = []
    for s in skills:
        s_l = s.lower()
        if s_l in job_keywords or any(s_l in kw or kw in s_l for kw in job_keywords):
            matched.append(s)
        else:
            unmatched.append(s)
    return matched[:limit], unmatched[:limit]


class ResumeAgent:
    def __init__(self, event_callback=None):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.event_callback = event_callback

    async def _emit(self, event_type: str, message: str, data: dict = {}):
        event = AgentEvent(event_type=event_type, agent_name="resume_agent", message=message, data=data)
        log.info("resume_agent.event", message=message)
        if self.event_callback:
            await self.event_callback(event)

    async def _tailor_cv_for_job(
        self,
        raw_cv: str,
        job: dict,
        selected_projects: list[dict] | None = None,
        matched_skills: list[str] | None = None,
    ) -> str:
        """Generate a tailored CV for a specific job."""
        selected_projects = selected_projects or []
        matched_skills = matched_skills or []
        projects_block = "\n".join(
            [f"- {p.get('title', 'Project')}: {p.get('summary', '')[:250]}" for p in selected_projects]
        ) or "- No explicit project entries found in uploaded CV"

        messages = [
            SystemMessage(content=CV_TAILORING_SYSTEM),
            HumanMessage(content=f"""
Candidate's Current CV:
{raw_cv}

Most Relevant Skills from Uploaded CV:
{", ".join(matched_skills) if matched_skills else "Not detected"}

Most Relevant Projects from Uploaded CV:
{projects_block}

---

Target Job Description:
Title: {job['title']}
Company: {job['company']}
Location: {job.get('location', 'N/A')}

{job.get('description', '')}

Requirements:
{chr(10).join(f"- {r}" for r in job.get('requirements', []))}

Please tailor this candidate's CV specifically for this role.
""")
        ]
        response = await self.llm.ainvoke(messages)
        return response.content

    def _fallback_cv_for_job(
        self,
        raw_cv: str,
        job: dict,
        selected_projects: list[dict] | None = None,
        matched_skills: list[str] | None = None,
        unmatched_skills: list[str] | None = None,
    ) -> str:
        """Deterministic CV fallback when LLM is unavailable."""
        selected_projects = selected_projects or []
        matched_skills = matched_skills or []
        unmatched_skills = unmatched_skills or []
        snippet = (raw_cv or "").strip()
        if len(snippet) > 1800:
            snippet = snippet[:1800]
        reqs = job.get("requirements", []) or []
        skill_lines = "\n".join(f"- {s}" for s in (matched_skills[:12] or reqs[:12])) if (matched_skills or reqs) else "- Adapt to role requirements"
        project_lines = "\n".join(
            f"- **{p.get('title', 'Project')}**: {p.get('summary', '')[:240]}"
            for p in selected_projects[:4]
        ) or "- No structured project section found in uploaded CV."
        additional_skills = "\n".join(f"- {s}" for s in unmatched_skills[:8]) or "- Available on request"
        return (
            f"# Candidate\n"
            f"**Target Role:** {job.get('title', 'Role')} at {job.get('company', 'Company')}\n\n"
            f"## Professional Summary\n"
            f"Candidate profile tailored for {job.get('title', 'the role')} with emphasis on relevant skills and responsibilities.\n\n"
            f"## Key Skills\n"
            f"{skill_lines}\n\n"
            f"## Selected Projects (From Uploaded CV)\n"
            f"{project_lines}\n\n"
            f"## Experience\n"
            f"{snippet or 'Experience details unavailable. Please upload CV for richer tailoring.'}\n\n"
            f"## Additional Skills\n"
            f"{additional_skills}\n\n"
            f"## Education\n"
            f"Available on request.\n"
        )

    async def _score_cv_ats(self, cv_content: str, job: dict) -> dict:
        """Score the tailored CV against the job description."""
        import json
        messages = [
            SystemMessage(content=ATS_SCORE_PROMPT.format(
                jd=f"{job['title']}\n{job.get('description', '')[:1000]}",
                cv=cv_content[:2000]
            ))
        ]
        try:
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
        except Exception:
            return {"ats_score": 0.7, "keywords_matched": [], "keywords_missing": [], "improvements": []}

    async def edit_cv_section(
        self,
        cv_id: str,
        user_id: str,
        instruction: str,
        section: str | None = None,
        direct_content: str | None = None,
    ) -> dict:
        """
        Edit a specific section of a CV based on user instruction or direct edit.
        This is the HITL editing feature.
        """
        # Load existing CV
        cvs = await long_term.get_cvs(user_id)
        cv = next((c for c in cvs if c["id"] == cv_id), None)
        if not cv:
            return {"error": "CV not found"}

        if direct_content:
            # User edited directly in the editor — just save
            new_content = direct_content
        else:
            # User asked the chatbot to edit a section
            current_markdown = cv.get("content_markdown", "")
            job_context = f"Job: {cv.get('data', {})}"

            messages = [
                SystemMessage(content=CV_SECTION_EDIT_SYSTEM.format(
                    section_content=current_markdown if not section else f"[{section} section]:\n{current_markdown}",
                    job_context=job_context
                )),
                HumanMessage(content=instruction)
            ]
            response = await self.llm.ainvoke(messages)
            
            if section:
                # Replace just the section
                import re
                section_pattern = rf"## {section}.*?(?=\n## |\Z)"
                new_section = response.content
                new_content = re.sub(section_pattern, new_section, current_markdown, flags=re.DOTALL | re.IGNORECASE)
                if new_content == current_markdown:  # section not found, append
                    new_content = current_markdown + f"\n\n## {section}\n{new_section}"
            else:
                new_content = response.content

        new_html = _cv_markdown_to_html(new_content)
        await long_term.update_cv_status(cv_id, CVStatus.EDITING, {"markdown": new_content, "html": new_html})

        return {"cv_id": cv_id, "content_markdown": new_content, "content_html": new_html}

    @trace_agent("resume_agent")
    async def run(self, user_id: str, workflow_id: str) -> dict[str, Any]:
        """
        Tailor CVs for all shortlisted jobs.
        Creates one tailored CV per job, saves to DB, triggers HITL.
        """
        # Load shortlisted jobs
        jobs = await long_term.get_jobs(user_id, status="shortlisted")
        if not jobs:
            # Fall back to top-scored fetched jobs
            jobs = await long_term.get_jobs(user_id)
            jobs = jobs[:10]  # Take top 10

        if not jobs:
            return {"total": 0, "cvs": []}

        # Load user's raw CV
        user_profile = await long_term.get_user_profile(user_id) or {}
        raw_cv = user_profile.get("data", {})
        if isinstance(raw_cv, str):
            import json
            try:
                raw_cv = json.loads(raw_cv)
            except Exception:
                raw_cv = {}
        raw_cv_text = raw_cv.get("raw_cv_text", "") if isinstance(raw_cv, dict) else ""
        parsed_skills = raw_cv.get("parsed_skills", []) if isinstance(raw_cv, dict) else []
        parsed_projects = raw_cv.get("parsed_projects", []) if isinstance(raw_cv, dict) else []
        if not isinstance(parsed_skills, list):
            parsed_skills = []
        if not isinstance(parsed_projects, list):
            parsed_projects = []

        if not raw_cv_text:
            await self._emit("agent_progress", "No CV uploaded. Using profile data only.", {"workflow_id": workflow_id})
            raw_cv_text = f"Name: {user_profile.get('name', 'Candidate')}\nEmail: {user_profile.get('email', '')}"
        else:
            await self._emit(
                "agent_progress",
                f"Using uploaded CV context ({len(parsed_projects)} projects, {len(parsed_skills)} skills extracted).",
                {"workflow_id": workflow_id},
            )

        cv_ids = []
        for i, job in enumerate(jobs):
            await self._emit("agent_progress",
                           f"Tailoring CV {i+1}/{len(jobs)}: {job['title']} at {job['company']}...",
                           {"workflow_id": workflow_id, "progress": (i+1)/len(jobs)})

            try:
                selected_projects = _select_relevant_projects(parsed_projects, job, limit=4)
                matched_skills, unmatched_skills = _match_skills_to_job(parsed_skills, job)
                try:
                    tailored_content = await self._tailor_cv_for_job(
                        raw_cv_text,
                        job,
                        selected_projects=selected_projects,
                        matched_skills=matched_skills,
                    )
                except Exception as e:
                    log.warning("resume_agent.tailor_fallback", job_id=job["id"], error=str(e))
                    tailored_content = self._fallback_cv_for_job(
                        raw_cv_text,
                        job,
                        selected_projects=selected_projects,
                        matched_skills=matched_skills,
                        unmatched_skills=unmatched_skills,
                    )

                ats_data = await self._score_cv_ats(tailored_content, job)
                cv_html = _cv_markdown_to_html(tailored_content)

                cv = {
                    "id": str(uuid.uuid4()),
                    "job_id": job["id"],
                    "job_title": job["title"],
                    "company": job["company"],
                    "content_markdown": tailored_content,
                    "content_html": cv_html,
                    "ats_score": ats_data.get("ats_score", 0.7),
                    "keywords_matched": ats_data.get("keywords_matched", []),
                    "keywords_missing": ats_data.get("keywords_missing", []),
                    "status": CVStatus.PENDING_APPROVAL,
                    "version": 1,
                    "data": {
                        "selected_projects": selected_projects,
                        "matched_skills": matched_skills,
                    },
                }

                await long_term.save_cv(cv, user_id)
                cv_ids.append(cv["id"])

            except Exception as e:
                log.error("resume_agent.cv_error", job_id=job["id"], error=str(e))

        return {"total": len(cv_ids), "cv_ids": cv_ids}
