"""
memory/store.py
Short-term memory (Redis or in-memory) + Long-term memory (PostgreSQL or in-memory).
Falls back gracefully to in-memory dicts when databases are unavailable (local development).
"""
import json
import os
from datetime import datetime
from typing import Any, Optional
import structlog

log = structlog.get_logger()

# ─────────────────────────────────────────────
# Short-term Memory — Redis (with in-memory fallback)
# ─────────────────────────────────────────────

class InMemoryCache:
    """Simple in-memory dict cache as fallback when Redis is unavailable."""
    def __init__(self):
        self._data: dict = {}
        self._lists: dict = {}

    async def set(self, key: str, value: Any, ttl: int = 3600):
        self._data[key] = json.dumps(value, default=str)

    async def get(self, key: str) -> Any | None:
        raw = self._data.get(key)
        return json.loads(raw) if raw else None

    async def delete(self, key: str):
        self._data.pop(key, None)

    async def rpush(self, key: str, value: str):
        self._lists.setdefault(key, []).append(value)

    async def ltrim(self, key: str, start: int, end: int):
        lst = self._lists.get(key, [])
        if end == -1:
            end = len(lst)
        self._lists[key] = lst[max(0, start):end + 1]

    async def lrange(self, key: str, start: int, end: int) -> list:
        lst = self._lists.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start:end + 1]

    async def aclose(self):
        pass


class ShortTermMemory:
    """Redis-backed (or in-memory fallback) short-term memory."""

    def __init__(self):
        self._client = None
        self._fallback = InMemoryCache()

    async def connect(self):
        try:
            import redis.asyncio as aioredis
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = await aioredis.from_url(url, decode_responses=True)
            await client.ping()
            self._client = client
            log.info("memory.redis", status="connected", url=url)
        except Exception as e:
            log.warning("memory.redis", status="unavailable — using in-memory fallback", error=str(e))
            self._client = self._fallback

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    @property
    def client(self):
        return self._client or self._fallback

    async def set(self, key: str, value: Any, ttl: int = 3600):
        data = json.dumps(value, default=str)
        try:
            await self.client.setex(key, ttl, data)
        except AttributeError:
            await self.client.set(key, value, ttl=ttl)

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self.client.get(key)
        except Exception:
            raw = None
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    async def delete(self, key: str):
        await self.client.delete(key)

    async def append_to_list(self, key: str, value: Any, max_len: int = 100):
        data = json.dumps(value, default=str)
        await self.client.rpush(key, data)
        await self.client.ltrim(key, -max_len, -1)

    async def get_list(self, key: str) -> list[Any]:
        items = await self.client.lrange(key, 0, -1)
        result = []
        for i in items:
            try:
                result.append(json.loads(i))
            except Exception:
                result.append(i)
        return result

    # ── Workflow State ────────────────────────

    async def set_workflow_status(self, workflow_id: str, status: dict):
        await self.set(f"workflow:{workflow_id}", status, ttl=86400)

    async def get_workflow_status(self, workflow_id: str) -> dict | None:
        return await self.get(f"workflow:{workflow_id}")

    async def set_current_agent(self, workflow_id: str, agent_name: str | None):
        await self.set(f"workflow:{workflow_id}:current_agent", agent_name, ttl=86400)

    async def get_current_agent(self, workflow_id: str) -> str | None:
        return await self.get(f"workflow:{workflow_id}:current_agent")

    # ── Conversation History ──────────────────

    async def add_message(self, session_id: str, message: dict):
        await self.append_to_list(f"chat:{session_id}", message, max_len=200)

    async def get_messages(self, session_id: str) -> list[dict]:
        return await self.get_list(f"chat:{session_id}")

    # ── HITL Checkpoints ─────────────────────

    async def set_hitl_checkpoint(self, checkpoint_id: str, data: dict):
        await self.set(f"hitl:{checkpoint_id}", data, ttl=86400 * 7)

    async def get_hitl_checkpoint(self, checkpoint_id: str) -> dict | None:
        return await self.get(f"hitl:{checkpoint_id}")

    async def resolve_hitl_checkpoint(self, checkpoint_id: str, approved: bool):
        data = await self.get_hitl_checkpoint(checkpoint_id)
        if data:
            data["approved"] = approved
            data["resolved_at"] = datetime.utcnow().isoformat()
            await self.set(f"hitl:{checkpoint_id}", data, ttl=3600)


# ─────────────────────────────────────────────
# Long-term Memory — PostgreSQL (with in-memory fallback)
# ─────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES user_profiles(id),
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    description TEXT,
    source_platform TEXT,
    source_url TEXT,
    match_score FLOAT DEFAULT 0,
    status TEXT DEFAULT 'fetched',
    hr_email TEXT,
    data JSONB NOT NULL DEFAULT '{}',
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tailored_cvs (
    id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(id),
    user_id TEXT REFERENCES user_profiles(id),
    content_markdown TEXT,
    content_html TEXT,
    ats_score FLOAT DEFAULT 0,
    status TEXT DEFAULT 'pending_approval',
    version INTEGER DEFAULT 1,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS email_drafts (
    id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(id),
    cv_id TEXT REFERENCES tailored_cvs(id),
    user_id TEXT REFERENCES user_profiles(id),
    hr_email TEXT NOT NULL,
    subject TEXT,
    body TEXT,
    cover_letter TEXT,
    status TEXT DEFAULT 'pending_approval',
    sent_at TIMESTAMPTZ,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_traces (
    id TEXT PRIMARY KEY,
    workflow_id TEXT,
    agent_name TEXT,
    action TEXT,
    input_data JSONB,
    output_data JSONB,
    duration_ms INTEGER,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_cvs_job ON tailored_cvs(job_id);
CREATE INDEX IF NOT EXISTS idx_emails_status ON email_drafts(status);
CREATE INDEX IF NOT EXISTS idx_traces_workflow ON agent_traces(workflow_id);
"""


class InMemoryDB:
    """Simple in-memory database fallback when PostgreSQL is unavailable."""
    def __init__(self):
        self.profiles: dict[str, dict] = {}
        self.jobs: list[dict] = []
        self.cvs: list[dict] = []
        self.email_drafts: list[dict] = []
        self.traces: list[dict] = []


class LongTermMemory:
    """PostgreSQL-backed (or in-memory fallback) persistent memory."""

    def __init__(self):
        self._pool = None
        self._fallback: InMemoryDB | None = None

    async def connect(self):
        try:
            import asyncpg
            url = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/candidates_fte")
            url = url.replace("postgresql+asyncpg://", "postgresql://")
            self._pool = await asyncpg.create_pool(url, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            log.info("memory.postgres", status="connected")
        except Exception as e:
            log.warning("memory.postgres", status="unavailable — using in-memory fallback", error=str(e))
            self._fallback = InMemoryDB()

    async def disconnect(self):
        if self._pool:
            await self._pool.close()

    @property
    def using_fallback(self) -> bool:
        return self._fallback is not None

    # ── User Profile ──────────────────────────

    async def upsert_user_profile(self, profile: dict) -> dict:
        if self.using_fallback:
            self._fallback.profiles[profile["id"]] = profile
            return profile
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO user_profiles (id, name, email, data)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (id) DO UPDATE
                   SET name=$2, email=$3, data=$4, updated_at=NOW()
                   RETURNING *""",
                profile["id"], profile["name"], profile["email"],
                json.dumps(profile)
            )
        return dict(row)

    async def get_user_profile(self, user_id: str) -> dict | None:
        if self.using_fallback:
            return self._fallback.profiles.get(user_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM user_profiles WHERE id=$1", user_id)
        return dict(row) if row else None

    # ── Jobs ──────────────────────────────────

    async def save_jobs(self, jobs: list[dict], user_id: str):
        valid_jobs = [j for j in jobs if j.get("id") and j.get("title") and j.get("company")]
        if len(valid_jobs) != len(jobs):
            log.warning("memory.save_jobs", skipped=len(jobs) - len(valid_jobs), reason="missing required job fields")

        if self.using_fallback:
            existing_ids = {j.get("id") for j in self._fallback.jobs if j.get("user_id") == user_id and j.get("id")}
            for job in valid_jobs:
                job["user_id"] = user_id
                if job.get("id") not in existing_ids:
                    self._fallback.jobs.append(job)
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO jobs (id, user_id, title, company, location, description,
                   source_platform, source_url, match_score, status, hr_email, data)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                   ON CONFLICT (id) DO UPDATE SET match_score=EXCLUDED.match_score,
                   status=EXCLUDED.status, data=EXCLUDED.data""",
                [(j.get("id"), user_id, j.get("title"), j.get("company"), j.get("location",""),
                  j.get("description",""), j.get("source_platform",""), j.get("source_url",""),
                  j.get("match_score", 0), j.get("status","fetched"),
                  j.get("hr_email"), json.dumps(j)) for j in valid_jobs]
            )

    async def get_jobs(self, user_id: str, status: str | None = None) -> list[dict]:
        if self.using_fallback:
            jobs = [j for j in self._fallback.jobs if j.get("user_id") == user_id]
            if status:
                jobs = [j for j in jobs if j.get("status") == status]
            return sorted(jobs, key=lambda j: j.get("match_score", 0), reverse=True)
        async with self._pool.acquire() as conn:
            if status:
                rows = await conn.fetch("SELECT * FROM jobs WHERE user_id=$1 AND status=$2 ORDER BY match_score DESC", user_id, status)
            else:
                rows = await conn.fetch("SELECT * FROM jobs WHERE user_id=$1 ORDER BY match_score DESC", user_id)
        return [dict(r) for r in rows]

    async def update_job_status(self, job_id: str, status: str, hr_email: str | None = None):
        if self.using_fallback:
            for job in self._fallback.jobs:
                if job["id"] == job_id:
                    job["status"] = status
                    if hr_email:
                        job["hr_email"] = hr_email
            return
        async with self._pool.acquire() as conn:
            if hr_email:
                await conn.execute("UPDATE jobs SET status=$1, hr_email=$2 WHERE id=$3", status, hr_email, job_id)
            else:
                await conn.execute("UPDATE jobs SET status=$1 WHERE id=$2", status, job_id)

    # ── CVs ───────────────────────────────────

    async def save_cv(self, cv: dict, user_id: str):
        if self.using_fallback:
            cv["user_id"] = user_id
            self._fallback.cvs.append(cv)
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO tailored_cvs (id, job_id, user_id, content_markdown, content_html,
                   ats_score, status, version, data)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                   ON CONFLICT (id) DO UPDATE
                   SET content_markdown=$4, content_html=$5, ats_score=$6,
                   status=$7, version=$8, data=$9, updated_at=NOW()""",
                cv["id"], cv["job_id"], user_id, cv["content_markdown"],
                cv.get("content_html",""), cv.get("ats_score",0), cv.get("status","pending_approval"),
                cv.get("version",1), json.dumps(cv)
            )

    async def get_cvs(self, user_id: str, status: str | None = None) -> list[dict]:
        if self.using_fallback:
            cvs = [c for c in self._fallback.cvs if c.get("user_id") == user_id]
            if status:
                cvs = [c for c in cvs if c.get("status") == status]
            return cvs
        async with self._pool.acquire() as conn:
            if status:
                rows = await conn.fetch("SELECT * FROM tailored_cvs WHERE user_id=$1 AND status=$2", user_id, status)
            else:
                rows = await conn.fetch("SELECT * FROM tailored_cvs WHERE user_id=$1 ORDER BY created_at DESC", user_id)
        return [dict(r) for r in rows]

    async def update_cv_status(self, cv_id: str, status: str, content: dict | None = None):
        if self.using_fallback:
            for cv in self._fallback.cvs:
                if cv["id"] == cv_id:
                    cv["status"] = status
                    if content:
                        cv["content_markdown"] = content.get("markdown", cv.get("content_markdown",""))
                        cv["content_html"] = content.get("html", cv.get("content_html",""))
            return
        async with self._pool.acquire() as conn:
            if content:
                await conn.execute(
                    "UPDATE tailored_cvs SET status=$1, content_markdown=$2, content_html=$3, updated_at=NOW() WHERE id=$4",
                    status, content.get("markdown",""), content.get("html",""), cv_id
                )
            else:
                await conn.execute("UPDATE tailored_cvs SET status=$1, updated_at=NOW() WHERE id=$2", status, cv_id)

    # ── Email Drafts ──────────────────────────

    async def save_email_draft(self, draft: dict, user_id: str):
        if self.using_fallback:
            draft["user_id"] = user_id
            # Update existing or append
            for i, d in enumerate(self._fallback.email_drafts):
                if d["id"] == draft["id"]:
                    self._fallback.email_drafts[i] = draft
                    return
            self._fallback.email_drafts.append(draft)
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO email_drafts (id, job_id, cv_id, user_id, hr_email, subject, body, cover_letter, status, data)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                   ON CONFLICT (id) DO UPDATE
                   SET subject=$6, body=$7, cover_letter=$8, status=$9, data=$10""",
                draft["id"], draft["job_id"], draft["cv_id"], user_id,
                draft["hr_email"], draft["subject"], draft["body"],
                draft["cover_letter"], draft.get("status","pending_approval"), json.dumps(draft)
            )

    async def get_email_drafts(self, user_id: str, status: str | None = None) -> list[dict]:
        if self.using_fallback:
            drafts = [d for d in self._fallback.email_drafts if d.get("user_id") == user_id]
            if status:
                drafts = [d for d in drafts if d.get("status") == status]
            return drafts
        async with self._pool.acquire() as conn:
            if status:
                rows = await conn.fetch("SELECT * FROM email_drafts WHERE user_id=$1 AND status=$2", user_id, status)
            else:
                rows = await conn.fetch("SELECT * FROM email_drafts WHERE user_id=$1 ORDER BY created_at DESC", user_id)
        return [dict(r) for r in rows]

    async def update_email_status(self, email_id: str, status: str, sent_at: datetime | None = None):
        if self.using_fallback:
            for draft in self._fallback.email_drafts:
                if draft["id"] == email_id:
                    draft["status"] = status
                    if sent_at:
                        draft["sent_at"] = sent_at.isoformat()
            return
        async with self._pool.acquire() as conn:
            if sent_at:
                await conn.execute("UPDATE email_drafts SET status=$1, sent_at=$2 WHERE id=$3", status, sent_at, email_id)
            else:
                await conn.execute("UPDATE email_drafts SET status=$1 WHERE id=$2", status, email_id)

    # ── Agent Traces ──────────────────────────

    async def log_trace(self, trace: dict):
        if self.using_fallback:
            self._fallback.traces.append(trace)
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO agent_traces (id, workflow_id, agent_name, action, input_data,
                   output_data, duration_ms, status)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                trace["id"], trace.get("workflow_id"), trace["agent_name"],
                trace["action"], json.dumps(trace.get("input",{})),
                json.dumps(trace.get("output",{})), trace.get("duration_ms"),
                trace.get("status","ok")
            )

    async def get_traces(self, workflow_id: str) -> list[dict]:
        if self.using_fallback:
            return [t for t in self._fallback.traces if t.get("workflow_id") == workflow_id]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM agent_traces WHERE workflow_id=$1 ORDER BY created_at ASC", workflow_id
            )
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Singleton instances
# ─────────────────────────────────────────────

short_term = ShortTermMemory()
long_term = LongTermMemory()


async def init_memory():
    await short_term.connect()
    await long_term.connect()
    log.info("memory", status="all stores connected")


async def close_memory():
    await short_term.disconnect()
    await long_term.disconnect()
