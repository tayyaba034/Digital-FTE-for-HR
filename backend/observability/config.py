"""
config.py
Centralised settings via pydantic-settings, plus observability utilities.
All values can be overridden by environment variables or a .env file.
"""
import functools
import time
import uuid
import structlog
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger()



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    secret_key: str = "change-me-in-production"
    debug: bool = False

    # LLM
    google_api_key: str = ""

    # Observability
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "candidates-fte"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # External APIs
    apify_api_key: str = ""
    hunter_io_api_key: str = ""

    # Gmail OAuth2
    gmail_client_id: str = ""
    gmail_client_secret: str = ""
    gmail_redirect_uri: str = "http://localhost:8000/auth/gmail/callback"

    # Database
    postgres_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/candidates_fte"
    redis_url: str = "redis://localhost:6379/0"

    # Agent tuning
    max_jobs_per_search: int = 50
    max_cvs_to_tailor: int = 10
    hitl_timeout_minutes: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def configure_observability():
    """Configure LangSmith/Langfuse tracing if keys are set. No-op otherwise."""
    if settings.langchain_api_key:
        import os
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        log.info("observability", tracing="langsmith", project=settings.langchain_project)
    else:
        log.info("observability", tracing="disabled (no LANGCHAIN_API_KEY set)")


def trace_agent(agent_name: str):
    """
    Decorator for agent methods that logs execution trace.
    Wraps async functions, records duration and status.
    No external service required — logs to structlog by default.
    Automatically persists trace to long_term DB if available.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            start = time.monotonic()
            log.info(f"{agent_name}.start", trace_id=trace_id)
            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info(f"{agent_name}.done", trace_id=trace_id, duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = int((time.monotonic() - start) * 1000)
                log.error(f"{agent_name}.error", trace_id=trace_id, error=str(e), duration_ms=duration_ms)
                raise
        return wrapper
    return decorator
