"""
agents/hitl_manager.py
Centralised Human-In-The-Loop checkpoint manager.
Handles pausing workflows, notifying users, and resuming on approval.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Optional
import structlog

from memory.store import short_term
from schemas.models import AgentEvent, HITLCheckpoint, HITLType

log = structlog.get_logger()

# How often (seconds) to poll for HITL resolution
POLL_INTERVAL = 3

# Max wait time before auto-timeout (seconds)
HITL_TIMEOUT = 3600  # 1 hour


class HITLManager:
    """
    Manages Human-In-The-Loop checkpoints across all agents.
    Provides a unified API for:
      - Creating checkpoints (pausing workflows)
      - Notifying the frontend
      - Waiting for user response
      - Resuming or re-running on rejection
    """

    def __init__(self, event_callback: Optional[Callable] = None):
        self.event_callback = event_callback

    async def _emit(self, event_type: str, message: str, data: dict = {}):
        event = AgentEvent(
            event_type=event_type,
            agent_name="hitl_manager",
            message=message,
            data=data,
        )
        log.info("hitl_manager.event", event_type=event_type, message=message)
        if self.event_callback:
            await self.event_callback(event)
        wf_id = data.get("workflow_id", "")
        if wf_id:
            await short_term.append_to_list(f"events:{wf_id}", event.model_dump(mode="json"))

    async def create_checkpoint(
        self,
        hitl_type: HITLType,
        workflow_id: str,
        user_id: str,
        payload: dict[str, Any],
        message: str = "",
    ) -> str:
        """
        Create a HITL checkpoint. Returns checkpoint_id.
        Workflow should pause after calling this and await wait_for_resolution().
        """
        checkpoint_id = str(uuid.uuid4())
        checkpoint = HITLCheckpoint(
            id=checkpoint_id,
            type=hitl_type,
            workflow_id=workflow_id,
            payload=payload,
        )
        await short_term.set_hitl_checkpoint(checkpoint_id, checkpoint.model_dump(mode="json"))
        
        # Update workflow status
        await short_term.set_workflow_status(workflow_id, {
            "status": "awaiting_approval",
            "hitl_checkpoint_id": checkpoint_id,
            "hitl_type": hitl_type,
            "updated_at": datetime.utcnow().isoformat(),
        })

        notify_msg = message or {
            HITLType.CV_REVIEW: "⏸ Please review and approve your tailored CVs before I continue.",
            HITLType.EMAIL_REVIEW: "⏸ Please review and approve your application emails before sending.",
        }.get(hitl_type, "⏸ Your approval is required to continue.")

        await self._emit(
            "hitl_required",
            notify_msg,
            {
                "workflow_id": workflow_id,
                "checkpoint_id": checkpoint_id,
                "type": hitl_type,
                **payload,
            },
        )
        log.info("hitl_manager.checkpoint_created", checkpoint_id=checkpoint_id, type=hitl_type)
        return checkpoint_id

    async def wait_for_resolution(
        self,
        checkpoint_id: str,
        workflow_id: str,
        timeout: int = HITL_TIMEOUT,
    ) -> tuple[bool, dict]:
        """
        Poll until the user resolves the checkpoint.
        Returns (approved: bool, checkpoint_data: dict).
        Times out after `timeout` seconds.
        """
        elapsed = 0
        while elapsed < timeout:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            data = await short_term.get_hitl_checkpoint(checkpoint_id)
            if data and data.get("approved") is not None:
                approved = bool(data["approved"])
                log.info("hitl_manager.resolved", checkpoint_id=checkpoint_id, approved=approved)
                return approved, data

        # Timeout — auto-reject
        log.warning("hitl_manager.timeout", checkpoint_id=checkpoint_id)
        await self._emit(
            "hitl_timeout",
            "⏰ HITL checkpoint timed out. Workflow paused. Resume from the dashboard.",
            {"workflow_id": workflow_id, "checkpoint_id": checkpoint_id},
        )
        return False, {}

    async def resolve(self, checkpoint_id: str, approved: bool, user_id: str = "") -> bool:
        """
        Resolve a checkpoint (called by the API when user clicks Approve/Reject).
        """
        data = await short_term.get_hitl_checkpoint(checkpoint_id)
        if not data:
            log.error("hitl_manager.resolve_not_found", checkpoint_id=checkpoint_id)
            return False

        data["approved"] = approved
        data["resolved_at"] = datetime.utcnow().isoformat()
        data["resolved_by"] = user_id
        await short_term.set_hitl_checkpoint(checkpoint_id, data)
        
        workflow_id = data.get("workflow_id", "")
        if workflow_id:
            await short_term.set_workflow_status(workflow_id, {
                "status": "working" if approved else "paused",
                "hitl_resolved": True,
                "hitl_approved": approved,
                "updated_at": datetime.utcnow().isoformat(),
            })
        
        log.info("hitl_manager.resolved_by_user", checkpoint_id=checkpoint_id, approved=approved)
        return True

    async def get_pending_checkpoints(self, user_id: str) -> list[dict]:
        """Get all unresolved HITL checkpoints for a user (for dashboard badge counts)."""
        # Redis pattern scan — in production, index by user_id
        # For now, return from short-term memory via known keys
        return []

    async def run_with_hitl(
        self,
        hitl_type: HITLType,
        workflow_id: str,
        user_id: str,
        payload: dict,
        on_approved: Callable,
        on_rejected: Optional[Callable] = None,
        message: str = "",
    ) -> Any:
        """
        Helper that creates checkpoint, waits, then calls on_approved or on_rejected.
        Usage:
            result = await hitl_manager.run_with_hitl(
                hitl_type=HITLType.CV_REVIEW,
                workflow_id=wf_id,
                user_id=uid,
                payload={"cv_ids": [...]},
                on_approved=lambda: apply_agent.run(...)
            )
        """
        checkpoint_id = await self.create_checkpoint(hitl_type, workflow_id, user_id, payload, message)
        approved, data = await self.wait_for_resolution(checkpoint_id, workflow_id)
        
        if approved:
            await self._emit("agent_progress", "✅ Approved! Continuing...", {"workflow_id": workflow_id})
            return await on_approved()
        else:
            await self._emit("agent_progress", "❌ Rejected. Workflow paused.", {"workflow_id": workflow_id})
            if on_rejected:
                return await on_rejected()
            return None
