"""
observability/config.py
LangSmith + Langfuse setup. Import this at app startup.
"""
import os
import functools
from typing import Any, Callable
import structlog
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

log = structlog.get_logger()

# ─────────────────────────────────────────────
# LangSmith — set via environment variables
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=candidates-fte
# These are auto-picked by LangChain when set.
# ─────────────────────────────────────────────

def configure_langsmith():
    """Ensure LangSmith env vars are set and tracing is enabled."""
    if not os.getenv("LANGCHAIN_API_KEY"):
        log.warning("observability.langsmith", status="LANGCHAIN_API_KEY not set — tracing disabled")
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "candidates-fte")
    log.info("observability.langsmith", status="enabled", project=os.getenv("LANGCHAIN_PROJECT"))


# ─────────────────────────────────────────────
# Langfuse — production monitoring
# ─────────────────────────────────────────────

_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    global _langfuse_client
    if _langfuse_client is None:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if pk and sk:
            _langfuse_client = Langfuse(public_key=pk, secret_key=sk, host=host)
            log.info("observability.langfuse", status="enabled", host=host)
        else:
            log.warning("observability.langfuse", status="disabled — LANGFUSE keys not set")
    return _langfuse_client


def trace_agent(agent_name: str):
    """
    Decorator to wrap any agent method with Langfuse tracing.
    
    Usage:
        @trace_agent("job_search_agent")
        async def run(self, ...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            lf = get_langfuse()
            if lf:
                trace = lf.trace(name=agent_name, metadata={"args": str(kwargs)})
                span = trace.span(name=f"{agent_name}.{fn.__name__}")
                try:
                    result = await fn(*args, **kwargs)
                    span.end(output=str(result)[:500])
                    return result
                except Exception as e:
                    span.end(level="ERROR", status_message=str(e))
                    raise
            else:
                return await fn(*args, **kwargs)
        return wrapper
    return decorator


def score_output(trace_id: str, name: str, value: float, comment: str = ""):
    """Record human feedback score (e.g., CV quality thumbs up/down)."""
    lf = get_langfuse()
    if lf:
        lf.score(trace_id=trace_id, name=name, value=value, comment=comment)


def configure_observability():
    """Call once at app startup."""
    configure_langsmith()
    get_langfuse()  # init client
