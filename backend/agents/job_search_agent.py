"""
agents/job_search_agent.py
Searches for jobs via Apify, deduplicates, and scores them against the user's profile.
"""
import uuid
from typing import Any
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from memory.store import long_term, short_term
from observability.config import trace_agent
from schemas.models import AgentEvent
from tools.apify_scraper import ApifyScraper

log = structlog.get_logger()

JOB_MATCHING_PROMPT = """You are a job-matching expert. Given a user's profile and a list of job listings, 
score each job from 0.0 to 1.0 based on how well it matches the candidate.

User Profile:
{profile}

Jobs to score (return JSON array with job IDs and scores):
{jobs}

For each job, consider: skill overlap, experience level match, location preferences, salary range.
Return ONLY a JSON array like: [{{"id": "abc123", "score": 0.87, "match_reasons": ["Python match", "Senior level"]}}, ...]"""

STOPWORDS = {
    "a", "an", "and", "for", "the", "to", "in", "with", "of", "on", "at", "role", "job", "jobs"
}

SOFTWARE_TITLE_HINTS = (
    "software engineer", "software developer", "developer", "backend engineer",
    "frontend engineer", "full stack", "full-stack", "platform engineer",
    "swe", "site reliability", "devops engineer", "ml engineer",
    "machine learning engineer", "data engineer", "python engineer",
    "java engineer", "react developer", "node.js developer"
)

NON_SOFTWARE_TITLE_HINTS = (
    "sales", "marketing", "account manager", "recruiter", "customer support",
    "nurse", "driver", "teacher", "hr ", "human resources", "accountant"
)


def _role_keywords(role: str) -> set[str]:
    base = {w for w in role.lower().replace("-", " ").split() if len(w) > 2 and w not in STOPWORDS}
    if "software" in base or "engineer" in base:
        base.update({"developer", "backend", "frontend", "fullstack", "full-stack", "swe"})
    if "data" in base:
        base.update({"analytics", "analyst", "scientist", "ml"})
    return base


def _is_software_role(role: str) -> bool:
    role_l = role.lower()
    return ("software" in role_l) or ("engineer" in role_l) or ("developer" in role_l)


def _is_relevant_job(role: str, job: dict) -> bool:
    keywords = _role_keywords(role)
    if not keywords:
        return True
    title = (job.get("title") or "").lower()
    description = (job.get("description") or "").lower()
    if _is_software_role(role):
        title_has_software_hint = any(term in title for term in SOFTWARE_TITLE_HINTS)
        title_has_non_software_hint = any(term in title for term in NON_SOFTWARE_TITLE_HINTS)
        if title_has_software_hint:
            return True
        if title_has_non_software_hint:
            return False
    title_hits = sum(1 for kw in keywords if kw in title)
    desc_hits = sum(1 for kw in keywords if kw in description)
    # Require title signal for role specificity; description alone is too noisy.
    if _is_software_role(role):
        return title_hits >= 1
    return title_hits >= 1 or desc_hits >= 2


def _query_variants(role: str) -> list[str]:
    role_l = role.lower().strip()
    variants = [role]
    if "software engineering" in role_l:
        variants.extend(["Software Engineer", "Software Developer"])
    if role_l in ("software engineer", "software engineering"):
        variants.extend(["Backend Engineer", "Full Stack Engineer"])
    if "developer" in role_l and "engineer" not in role_l:
        variants.append(role.replace("developer", "engineer"))

    seen: set[str] = set()
    deduped: list[str] = []
    for v in variants:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(v.strip())
    return deduped[:4]


class JobSearchAgent:
    def __init__(self, event_callback=None):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.scraper = ApifyScraper()
        self.event_callback = event_callback

    async def _emit(self, event_type: str, message: str, data: dict = {}):
        event = AgentEvent(event_type=event_type, agent_name="job_search_agent", message=message, data=data)
        log.info("job_search_agent.event", message=message)
        if self.event_callback:
            await self.event_callback(event)

    async def _score_jobs(self, jobs: list[dict], user_profile: dict) -> list[dict]:
        """Use Claude to score job matches."""
        if not jobs:
            return jobs

        profile_data = user_profile.get("data", {}) if isinstance(user_profile.get("data"), dict) else {}
        has_profile_signals = bool(
            user_profile.get("name")
            or user_profile.get("email")
            or profile_data.get("skills")
            or profile_data.get("parsed_skills")
            or profile_data.get("target_roles")
            or profile_data.get("experience_years")
            or profile_data.get("raw_cv_text")
        )
        if not has_profile_signals:
            for job in jobs:
                job["match_score"] = 0.5
                job["match_reasons"] = []
            return jobs

        # Slim down jobs for the prompt (save tokens)
        slim_jobs = [
            {"id": j["id"], "title": j["title"], "company": j["company"],
             "description": j.get("description", "")[:500],
             "requirements": j.get("requirements", [])}
            for j in jobs
        ]

        profile_summary = {
            "skills": user_profile.get("skills", []) or profile_data.get("parsed_skills", []),
            "experience_years": user_profile.get("experience_years"),
            "target_roles": user_profile.get("target_roles", []),
            "location": user_profile.get("location"),
        }

        import json
        messages = [
            SystemMessage(content=JOB_MATCHING_PROMPT.format(
                profile=json.dumps(profile_summary),
                jobs=json.dumps(slim_jobs)
            )),
            HumanMessage(content="Score all jobs and return JSON only."),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            scores = json.loads(response.content)
            if not isinstance(scores, list):
                scores = []

            score_map: dict[str, dict] = {}
            for s in scores:
                if not isinstance(s, dict):
                    continue
                sid = s.get("id") or s.get('"id"')
                if sid:
                    score_map[str(sid)] = s

            for job in jobs:
                match = score_map.get(job["id"], {})
                job["match_score"] = match.get("score", 0.5)
                job["match_reasons"] = match.get("match_reasons", [])

            # Sort by match score
            jobs.sort(key=lambda j: j.get("match_score", 0), reverse=True)
        except Exception as e:
            log.error("job_search_agent.scoring_error", error=str(e))
            for job in jobs:
                job["match_score"] = job.get("match_score", 0.5)
                job["match_reasons"] = job.get("match_reasons", [])

        return jobs

    @trace_agent("job_search_agent")
    async def run(
        self,
        user_id: str,
        workflow_id: str,
        role: str = "Software Engineer",
        locations: list[str] | None = None,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """
        Full job search pipeline:
        1. Scrape jobs from multiple platforms
        2. Deduplicate
        3. Score against user profile
        4. Save to DB
        """
        locations = locations or ["Remote"]

        await self._emit("agent_progress", f"Searching for '{role}' jobs in {', '.join(locations)}...",
                        {"workflow_id": workflow_id})

        # 1. Scrape
        jobs, duplicates_removed = await self.scraper.scrape_jobs(
            query=role,
            locations=locations,
            max_results=max_results,
        )
        all_scraped_jobs = list(jobs)

        # Retry with Remote if the initial location returns no jobs.
        if not jobs and [l.lower() for l in locations] != ["remote"]:
            await self._emit(
                "agent_progress",
                f"No jobs found for {', '.join(locations)}. Retrying with Remote...",
                {"workflow_id": workflow_id},
            )
            jobs, duplicates_removed = await self.scraper.scrape_jobs(
                query=role,
                locations=["Remote"],
                max_results=max_results,
            )
            all_scraped_jobs.extend(jobs)
        await self._emit("agent_progress", 
                        f"Found {len(jobs) + duplicates_removed} listings; {len(jobs)} unique after removing {duplicates_removed} duplicates",
                        {"workflow_id": workflow_id, "total_raw": len(jobs) + duplicates_removed,
                         "deduplicated": len(jobs), "removed": duplicates_removed})

        # 2. Load user profile for matching
        pre_filter_count = len(jobs)
        relevant_jobs = [j for j in jobs if _is_relevant_job(role, j)]
        filtered_out = pre_filter_count - len(relevant_jobs)
        if filtered_out > 0:
            await self._emit(
                "agent_progress",
                f"Filtered out {filtered_out} jobs that do not match '{role}'.",
                {"workflow_id": workflow_id, "filtered_out": filtered_out},
            )
        if relevant_jobs:
            jobs = relevant_jobs
        elif _is_software_role(role):
            await self._emit(
                "agent_progress",
                f"No strongly relevant '{role}' titles found. Retrying with software-role aliases...",
                {"workflow_id": workflow_id},
            )
            alias_queries = ["Software Engineer", "Software Developer", "Backend Engineer"]
            retried_jobs: list[dict] = []
            for alias in alias_queries:
                alias_jobs, _ = await self.scraper.scrape_jobs(
                    query=alias,
                    locations=locations,
                    max_results=max_results,
                )
                retried_jobs.extend(alias_jobs)
                all_scraped_jobs.extend(alias_jobs)
            dedup: dict[str, dict] = {}
            for job in retried_jobs:
                jid = job.get("id")
                if jid and jid not in dedup:
                    dedup[jid] = job
            filtered_retry = [j for j in dedup.values() if _is_relevant_job(role, j)]
            if filtered_retry:
                jobs = filtered_retry
                await self._emit(
                    "agent_progress",
                    f"Recovered {len(jobs)} role-specific jobs after alias retry.",
                    {"workflow_id": workflow_id},
                )
            else:
                jobs = []
                await self._emit(
                    "agent_progress",
                    f"No role-specific matches found for '{role}'. Please refine role or location.",
                    {"workflow_id": workflow_id},
                )
        else:
            await self._emit(
                "agent_progress",
                f"No strongly relevant matches found for '{role}'. Keeping best available results.",
                {"workflow_id": workflow_id},
            )

        # If relevant set is very small, broaden with query variants as a second pass.
        if len(jobs) < 5:
            await self._emit(
                "agent_progress",
                f"Low results ({len(jobs)}). Expanding search with role variants...",
                {"workflow_id": workflow_id},
            )
            broadened, _ = await self.scraper.scrape_jobs(
                query=role,
                locations=locations,
                max_results=max_results,
                query_variants=_query_variants(role),
            )
            all_scraped_jobs.extend(broadened)
            merged: dict[str, dict] = {j.get("id"): j for j in jobs if j.get("id")}
            for j in broadened:
                jid = j.get("id")
                if jid and jid not in merged and _is_relevant_job(role, j):
                    merged[jid] = j
            jobs = list(merged.values())

        # Last-resort safety: never return 0 if we scraped anything at all.
        if not jobs and all_scraped_jobs:
            fallback_map: dict[str, dict] = {}
            for j in all_scraped_jobs:
                jid = j.get("id")
                if jid and jid not in fallback_map:
                    fallback_map[jid] = j
            jobs = list(fallback_map.values())[: max(5, min(20, max_results))]
            await self._emit(
                "agent_progress",
                f"Strict role filter produced 0 results; returning {len(jobs)} best available listings.",
                {"workflow_id": workflow_id},
            )

        # 2. Load user profile for matching
        user_profile = await long_term.get_user_profile(user_id) or {}

        # 3. Score
        await self._emit("agent_progress", "Scoring job matches against your profile...", {"workflow_id": workflow_id})
        jobs = await self._score_jobs(jobs, user_profile)

        # 4. Save to DB
        await long_term.save_jobs(jobs, user_id)
        await short_term.set(f"jobs:{user_id}:latest", {"count": len(jobs), "workflow_id": workflow_id})

        await self._emit("agent_done",
                        f"Saved {len(jobs)} jobs. Top match: {jobs[0]['title']} at {jobs[0]['company']} ({jobs[0].get('match_score', 0):.0%})" 
                        if jobs else "Search complete (no results found)",
                        {"workflow_id": workflow_id})

        return {
            "total": len(jobs),
            "duplicates_removed": duplicates_removed,
            "top_jobs": jobs[:5],
        }
