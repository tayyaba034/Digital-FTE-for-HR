"""
agents/interview_agent.py
Interview preparation: generates questions, runs mock interviews, skill gap analysis.
"""
import uuid
from typing import Any
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from memory.store import long_term, short_term
from observability.config import trace_agent
from schemas.models import AgentEvent

log = structlog.get_logger()

INTERVIEW_QUESTIONS_PROMPT = """You are an expert technical interviewer. Generate interview questions for this role.

Job: {title} at {company}
Description: {description}
Candidate Skills: {skills}

Generate 15 interview questions in this JSON format:
[
  {{
    "id": "q1",
    "category": "technical",
    "question": "...",
    "ideal_answer_hints": ["hint1", "hint2"],
    "difficulty": "medium"
  }},
  ...
]

Mix: 6 technical, 5 behavioral (STAR format), 4 situational.
Tailor to the specific company and role — no generic questions."""

MOCK_INTERVIEW_SYSTEM = """You are a professional interviewer at {company} conducting a {title} interview.

You are interviewing for this role:
{description}

Conduct a realistic, professional mock interview:
- Ask one question at a time
- After the candidate answers, provide brief feedback (what was good, what to improve)
- Then ask the next question
- Be encouraging but honest
- After 5 questions, provide an overall performance summary

Start by introducing yourself and asking the first question."""

SKILL_GAP_PROMPT = """Analyze the gap between this candidate's skills and the job requirements.

Candidate Skills: {candidate_skills}
Job Requirements: {job_requirements}

Return JSON:
{{
  "skill_gaps": ["Kubernetes", "Go language"],
  "strengths": ["Python", "System Design"],
  "priority_learning": [
    {{"skill": "Kubernetes", "resource": "Kubernetes for Developers course on Pluralsight", "time_weeks": 3}},
    ...
  ],
  "overall_readiness": 0.75
}}"""


class InterviewAgent:
    def __init__(self, event_callback=None):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        self.event_callback = event_callback

    async def _emit(self, event_type: str, message: str, data: dict = {}):
        event = AgentEvent(event_type=event_type, agent_name="interview_agent", message=message, data=data)
        log.info("interview_agent.event", message=message)
        if self.event_callback:
            await self.event_callback(event)

    def _fallback_questions(self, job: dict) -> list[dict]:
        title = job.get("title", "the role")
        company = job.get("company", "the company")
        items = [
            ("technical", f"Walk me through a project most relevant to {title}.", ["Architecture", "Tradeoffs", "Outcome"], "medium"),
            ("technical", "How do you debug performance bottlenecks in production?", ["Profiling", "Metrics", "Hypothesis-driven fixes"], "medium"),
            ("technical", "How would you design a scalable API for this product?", ["Data model", "Caching", "Failure handling"], "hard"),
            ("behavioral", f"Tell me about a time you handled conflicting priorities while delivering at {company}-level quality.", ["STAR format", "Impact", "What you learned"], "medium"),
            ("behavioral", "Describe a difficult team disagreement and how you resolved it.", ["Communication", "Decision process", "Outcome"], "medium"),
            ("situational", "If requirements are ambiguous, how do you proceed without blocking delivery?", ["Clarify assumptions", "Prototype", "Iterate"], "medium"),
        ]
        out: list[dict] = []
        for category, question, hints, difficulty in items:
            out.append({
                "id": str(uuid.uuid4()),
                "job_id": job.get("id"),
                "category": category,
                "question": question,
                "ideal_answer_hints": hints,
                "difficulty": difficulty,
            })
        return out

    async def generate_questions(self, job: dict, user_profile: dict) -> list[dict]:
        """Generate tailored interview questions for a job."""
        import json
        
        skills = []
        if isinstance(user_profile.get("data"), dict):
            skills = user_profile["data"].get("skills", [])

        messages = [
            SystemMessage(content=INTERVIEW_QUESTIONS_PROMPT.format(
                title=job["title"],
                company=job["company"],
                description=job.get("description", "")[:1000],
                skills=", ".join(skills[:20])
            ))
        ]
        try:
            response = await self.llm.ainvoke(messages)
            questions = json.loads(response.content)
            for q in questions:
                q["id"] = str(uuid.uuid4())
                q["job_id"] = job["id"]
            return questions
        except Exception as e:
            log.warning("interview_agent.questions_fallback", error=str(e), job_id=job.get("id"))
            return self._fallback_questions(job)

    async def mock_interview_chat(
        self,
        session_id: str,
        user_message: str,
        job: dict,
    ) -> str:
        """Handle a message in a mock interview session."""
        # Load conversation history
        history = await short_term.get_messages(f"interview:{session_id}")
        
        if not history:
            # Start of interview — build system message
            system = MOCK_INTERVIEW_SYSTEM.format(
                company=job["company"],
                title=job["title"],
                description=job.get("description", "")[:800]
            )
        else:
            system = history[0].get("content", "") if history and history[0]["role"] == "system" else ""

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        
        for msg in history:
            if msg["role"] == "system":
                continue
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_message))

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
        except Exception as e:
            log.warning("interview_agent.mock_chat_fallback", error=str(e), session_id=session_id)
            content = (
                "Thanks for your answer. You explained your approach clearly.\n\n"
                "Follow-up: describe one concrete tradeoff you made and how you measured success."
            )

        # Save to history
        await short_term.add_message(f"interview:{session_id}", {"role": "user", "content": user_message})
        await short_term.add_message(f"interview:{session_id}", {"role": "assistant", "content": content})

        return content

    async def analyze_skill_gaps(self, job: dict, user_profile: dict) -> dict:
        """Analyze skill gaps and suggest learning resources."""
        import json
        
        candidate_skills = []
        if isinstance(user_profile.get("data"), dict):
            candidate_skills = user_profile["data"].get("skills", [])

        messages = [
            SystemMessage(content=SKILL_GAP_PROMPT.format(
                candidate_skills=", ".join(candidate_skills),
                job_requirements=", ".join(job.get("requirements", []))
            ))
        ]
        try:
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
        except Exception as e:
            log.warning("interview_agent.skill_gap_fallback", error=str(e), job_id=job.get("id"))
            return {"skill_gaps": [], "strengths": candidate_skills, "priority_learning": [], "overall_readiness": 0.7}

    @trace_agent("interview_agent")
    async def run(self, user_id: str, workflow_id: str) -> dict[str, Any]:
        """Generate interview prep for all applied/shortlisted jobs."""
        jobs = await long_term.get_jobs(user_id, status="applied")
        if not jobs:
            jobs = await long_term.get_jobs(user_id)
            jobs = jobs[:5]

        user_profile = await long_term.get_user_profile(user_id) or {}
        prep_results = []

        for job in jobs:
            await self._emit("agent_progress",
                           f"Generating interview prep for {job['title']} at {job['company']}...",
                           {"workflow_id": workflow_id})

            questions = await self.generate_questions(job, user_profile)
            skill_gaps = await self.analyze_skill_gaps(job, user_profile)

            result = {
                "job_id": job["id"],
                "job_title": job["title"],
                "company": job["company"],
                "questions": questions,
                "skill_gaps": skill_gaps,
                "session_id": str(uuid.uuid4()),
            }
            prep_results.append(result)

            # Cache in Redis for quick retrieval
            await short_term.set(f"interview_prep:{job['id']}", result, ttl=86400 * 7)

        return {"total": len(prep_results), "preps": prep_results}
