"""
tools/apify_scraper.py
Job scraping via Apify API with deduplication logic.
"""
import hashlib
import os
import re
import asyncio
from typing import Any
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

APIFY_BASE_URL = "https://api.apify.com/v2"

# Apify actor IDs for each platform
ACTORS = {
    "linkedin": "curious_coder~linkedin-jobs-scraper",
    "indeed": "misceres~indeed-scraper",
    "glassdoor": "bebity~glassdoor-jobs-scraper",
    "upwork": "getdataforme~upwork-jobs-scraper",
}


def _job_fingerprint(title: str, company: str) -> str:
    """Create a canonical ID for deduplication based on title + company."""
    normalized = f"{title.lower().strip()}|{company.lower().strip()}"
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _deduplicate(jobs: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate jobs (same role at same company across platforms)."""
    seen: dict[str, dict] = {}
    for job in jobs:
        fp = _job_fingerprint(job["title"], job["company"])
        if fp not in seen:
            job["id"] = fp
            seen[fp] = job
        else:
            # Merge: keep the one with more info, add extra source
            existing = seen[fp]
            if "extra_sources" not in existing:
                existing["extra_sources"] = []
            existing["extra_sources"].append(job.get("source_platform"))
    
    duplicates_removed = len(jobs) - len(seen)
    return list(seen.values()), duplicates_removed


class ApifyScraper:
    def __init__(self):
        self.api_key = os.getenv("APIFY_API_KEY")
        if not self.api_key:
            log.warning("apify", status="APIFY_API_KEY not set — scraping will be mocked")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _run_actor(self, actor_id: str, input_data: dict) -> list[dict]:
        """Run an Apify actor and return results."""
        if not self.api_key:
            return self._mock_jobs(input_data.get("query", "Software Engineer"), actor_id)

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Start run
            resp = await client.post(
                f"{APIFY_BASE_URL}/acts/{actor_id}/runs",
                params={"token": self.api_key},
                json=input_data
            )
            resp.raise_for_status()
            run_id = resp.json()["data"]["id"]
            log.info("apify.actor_started", actor=actor_id, run_id=run_id)

            # Wait for completion (poll)
            max_polls = int(os.getenv("APIFY_MAX_POLLS", "15"))  # ~45s with default poll interval
            poll_interval = int(os.getenv("APIFY_POLL_INTERVAL_SECONDS", "3"))
            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)
                status_resp = await client.get(
                    f"{APIFY_BASE_URL}/actor-runs/{run_id}",
                    params={"token": self.api_key}
                )
                status = status_resp.json()["data"]["status"]
                if status == "SUCCEEDED":
                    break
                elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    raise RuntimeError(f"Apify actor {actor_id} failed with status: {status}")
            else:
                raise RuntimeError(f"Apify actor {actor_id} timed out waiting for completion")

            # Fetch results
            results_resp = await client.get(
                f"{APIFY_BASE_URL}/actor-runs/{run_id}/dataset/items",
                params={"token": self.api_key, "format": "json"}
            )
            results_resp.raise_for_status()
            return results_resp.json()

    @staticmethod
    def _safe_text(value: Any, fallback: str) -> str:
        text = str(value or "").strip()
        return text if text else fallback

    def _actor_registry(self) -> dict[str, str]:
        """Build actor map from defaults + environment overrides/extra sources."""
        actors = dict(ACTORS)

        env_overrides = {
            "linkedin": os.getenv("APIFY_ACTOR_LINKEDIN"),
            "indeed": os.getenv("APIFY_ACTOR_INDEED"),
            "glassdoor": os.getenv("APIFY_ACTOR_GLASSDOOR"),
            "upwork": os.getenv("APIFY_ACTOR_UPWORK"),
        }
        for platform, actor_id in env_overrides.items():
            if actor_id:
                actors[platform] = actor_id.strip()

        # Format: APIFY_EXTRA_ACTORS="ziprecruiter=owner~actor,monster=owner~actor"
        extra = os.getenv("APIFY_EXTRA_ACTORS", "").strip()
        if extra:
            for pair in extra.split(","):
                if "=" not in pair:
                    continue
                platform, actor_id = pair.split("=", 1)
                platform = platform.strip().lower()
                actor_id = actor_id.strip()
                if platform and actor_id:
                    actors[platform] = actor_id

        return actors

    def _build_input(self, platform: str, query: str, locations: list[str], max_items: int) -> dict:
        location = locations[0] if locations else "Remote"
        if platform == "linkedin":
            return {"keywords": query, "location": location, "maxResults": max_items}
        if platform == "indeed":
            return {"query": query, "location": location, "maxItems": max_items}
        if platform == "glassdoor":
            return {"keyword": query, "location": location, "maxItems": max_items}
        if platform == "upwork":
            return {"searchKeyword": query, "searchLocation": locations or [location], "maxLimit": max_items}
        # Generic fallback for extra actors
        return {"query": query, "keyword": query, "location": location, "maxItems": max_items}

    def _extract_first(self, raw: dict, keys: list[str]) -> str:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _normalize_generic(self, raw: dict, platform: str) -> dict:
        title = self._safe_text(
            self._extract_first(raw, ["title", "jobTitle", "positionName", "position", "role"]),
            "Software Engineer",
        )
        company = self._safe_text(
            self._extract_first(raw, ["company", "companyName", "employer", "organization", "company_name"]),
            "Unknown Company",
        )
        return {
            "title": title,
            "company": company,
            "location": self._extract_first(raw, ["location", "jobLocation", "city"]),
            "description": self._extract_first(raw, ["description", "jobDescription", "snippet", "summary"]),
            "source_url": self._extract_first(raw, ["url", "jobUrl", "jobListingUrl", "link"]),
            "source_platform": platform,
            "posted_date": self._extract_first(raw, ["postedAt", "datePosted", "posted", "date"]),
            "salary_range": self._extract_first(raw, ["salary", "salaryRange", "estimatedSalary"]),
            "job_type": self._extract_first(raw, ["jobType", "contractType", "employmentType"]),
            "requirements": [],
        }

    def _normalize_upwork(self, raw: dict) -> dict:
        title = self._safe_text(
            self._extract_first(raw, ["job_title", "title", "jobTitle", "titleText"]),
            "Freelance Role",
        )
        company = self._safe_text(
            self._extract_first(raw, ["client_name", "client", "buyer", "company"]),
            "Upwork Client",
        )
        desc = self._extract_first(raw, ["job_description", "description", "summary", "snippet"])
        skills = raw.get("skills") or raw.get("required_skills") or []
        if not isinstance(skills, list):
            skills = []
        requirements = [str(s).strip() for s in skills if str(s).strip()]
        return {
            "title": title,
            "company": company,
            "location": self._extract_first(raw, ["client_location", "location", "country"]),
            "description": desc,
            "source_url": self._extract_first(raw, ["job_url", "url", "link"]),
            "source_platform": "upwork",
            "posted_date": self._extract_first(raw, ["posted_time", "createdAt", "datePosted"]),
            "salary_range": self._extract_first(raw, ["budget", "amount", "hourlyRate"]),
            "job_type": self._extract_first(raw, ["job_type", "type", "engagement"]),
            "requirements": requirements,
        }

    def _normalize_linkedin(self, raw: dict) -> dict:
        title = self._safe_text(raw.get("title"), "Software Engineer")
        company = self._safe_text(raw.get("companyName"), "Unknown Company")
        return {
            "title": title,
            "company": company,
            "location": raw.get("location", ""),
            "description": raw.get("description", ""),
            "source_url": raw.get("jobUrl", ""),
            "source_platform": "linkedin",
            "posted_date": raw.get("postedAt"),
            "salary_range": raw.get("salary"),
            "job_type": raw.get("contractType"),
            "requirements": [],
        }

    def _normalize_indeed(self, raw: dict) -> dict:
        title = self._safe_text(raw.get("positionName"), "Software Engineer")
        company = self._safe_text(raw.get("company"), "Unknown Company")
        return {
            "title": title,
            "company": company,
            "location": raw.get("location", ""),
            "description": raw.get("description", ""),
            "source_url": raw.get("url", ""),
            "source_platform": "indeed",
            "posted_date": raw.get("datePosted"),
            "salary_range": raw.get("salary"),
            "job_type": raw.get("jobType"),
            "requirements": [],
        }

    def _normalize_glassdoor(self, raw: dict) -> dict:
        title = self._safe_text(raw.get("jobTitle"), "Software Engineer")
        company = self._safe_text(raw.get("employer", {}).get("name"), "Unknown Company")
        return {
            "title": title,
            "company": company,
            "location": raw.get("location", ""),
            "description": raw.get("jobDescription", ""),
            "source_url": raw.get("jobListingUrl", ""),
            "source_platform": "glassdoor",
            "posted_date": raw.get("discoveredAt"),
            "salary_range": raw.get("estimatedSalary"),
            "job_type": raw.get("jobType"),
            "requirements": [],
        }

    def _mock_jobs(self, query: str, actor_id: str) -> list[dict]:
        """Return mock data when API key is not set (for development)."""
        platform = "linkedin" if "linkedin" in actor_id else "indeed" if "indeed" in actor_id else "glassdoor"
        return [
            {
                "title": f"Senior {query}",
                "company": f"TechCorp {platform.title()}",
                "location": "Remote / Pakistan",
                "description": f"We are looking for a Senior {query} to join our growing team. "
                               "You will be responsible for designing scalable systems, "
                               "mentoring junior engineers, and shipping high-quality software.",
                "source_url": f"https://{platform}.com/jobs/mock-{platform}-1",
                "source_platform": platform,
                "salary_range": "PKR 5,000,000 - 8,000,000",
                "job_type": "Full-time",
                "requirements": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
            },
            {
                "title": f"Staff {query}",
                "company": f"StartupXYZ {platform.title()}",
                "location": "Karachi, Pakistan",
                "description": f"Join our mission-driven team as a Staff {query}. "
                               "We're building the next generation of developer tools.",
                "source_url": f"https://{platform}.com/jobs/mock-{platform}-2",
                "source_platform": platform,
                "salary_range": "PKR 7,000,000 - 10,000,000",
                "job_type": "Full-time",
                "requirements": ["Python", "LangChain", "React", "PostgreSQL"],
            }
        ]

    async def scrape_jobs(
        self,
        query: str,
        locations: list[str],
        max_results: int = 50,
        platforms: list[str] | None = None,
        query_variants: list[str] | None = None,
    ) -> tuple[list[dict], int]:
        """
        Scrape jobs from multiple platforms and return deduplicated results.
        Returns: (jobs, duplicates_removed)
        """
        actor_map = self._actor_registry()
        if platforms is None:
            platforms = [p for p in ["linkedin", "indeed", "glassdoor", "upwork"] if p in actor_map]
        if not platforms:
            platforms = list(actor_map.keys())

        variants = [query]
        if query_variants:
            for v in query_variants:
                v_clean = (v or "").strip()
                if v_clean and v_clean.lower() not in {x.lower() for x in variants}:
                    variants.append(v_clean)
        # Avoid too many actor runs while still broadening search.
        variants = variants[: int(os.getenv("APIFY_MAX_QUERY_VARIANTS", "2"))]

        async def scrape_platform(platform: str, variant_query: str) -> list[dict]:
            actor_id = actor_map.get(platform)
            if not actor_id:
                return []

            try:
                log.info("apify.scraping", platform=platform, query=variant_query)
                per_run = max(1, max_results // max(1, len(platforms) * len(variants)))
                input_data = self._build_input(platform, variant_query, locations, per_run)
                raw = await self._run_actor(actor_id, input_data)
                if platform == "linkedin":
                    normalized = [self._normalize_linkedin(r) for r in raw]
                elif platform == "indeed":
                    normalized = [self._normalize_indeed(r) for r in raw]
                elif platform == "glassdoor":
                    normalized = [self._normalize_glassdoor(r) for r in raw]
                elif platform == "upwork":
                    normalized = [self._normalize_upwork(r) for r in raw]
                else:
                    normalized = [self._normalize_generic(r, platform) for r in raw]

                log.info("apify.scraped", platform=platform, query=variant_query, count=len(normalized))
                return normalized

            except Exception as e:
                log.error("apify.scrape_error", platform=platform, query=variant_query, error=str(e))
                return []

        async def scrape_with_timeout(platform: str, variant_query: str) -> list[dict]:
            timeout_s = int(os.getenv("APIFY_PLATFORM_TIMEOUT_SECONDS", "55"))
            try:
                return await asyncio.wait_for(scrape_platform(platform, variant_query), timeout=timeout_s)
            except asyncio.TimeoutError:
                log.error("apify.scrape_timeout", platform=platform, query=variant_query, timeout_seconds=timeout_s)
                return []

        # Run platform+variant scrapes concurrently; return partial results quickly.
        tasks = [scrape_with_timeout(p, v) for p in platforms for v in variants]
        results = await asyncio.gather(*tasks)
        all_raw = [job for batch in results for job in batch]
        if not all_raw:
            log.warning(
                "apify.empty_results",
                message="All sources returned empty; falling back to synthetic seed jobs",
                query=query,
                locations=locations,
                platforms=platforms,
            )
            seeds: list[dict] = []
            for p in platforms:
                actor_id = actor_map.get(p, p)
                seeds.extend(self._mock_jobs(query, actor_id))
            all_raw = seeds
        jobs, duplicates_removed = _deduplicate(all_raw)
        log.info("apify.complete", total=len(jobs), duplicates_removed=duplicates_removed, platforms=platforms, variants=variants)
        return jobs, duplicates_removed
