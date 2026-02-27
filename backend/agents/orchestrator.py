"""
agents/orchestrator.py
Central Orchestrator Agent using LangGraph.
Reads user intent → activates the right sub-agents → manages workflow state.
"""
import uuid
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Literal
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from memory.store import short_term, long_term
from observability.config import trace_agent
from schemas.models import AgentEvent, AgentStatus, WorkflowState

log = structlog.get_logger()

# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class OrchestratorState(BaseModel):
    workflow_id: str
    user_id: str
    session_id: str
    raw_query: str
    intent: str = ""                      # parsed intent key
    agents_to_run: list[str] = []        # ordered list
    completed_agents: list[str] = []
    current_agent: str = ""
    agent_results: dict[str, Any] = {}
    hitl_required: bool = False
    hitl_checkpoint_id: str = ""
    error: str = ""
    status: str = AgentStatus.WORKING


# ─────────────────────────────────────────────
# Intent Detection
# ─────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are the Orchestrator of Candidates FTE — a Digital Full-Time Employee for job searching and applying.

Your job is to analyze the user's query and return a JSON object with:
1. "intent": one of ["full_pipeline", "job_search_only", "cv_only", "apply_only", "interview_only", "custom"]
2. "agents": ordered list of agents to activate, chosen from:
   - "job_search_agent"
   - "resume_agent" 
   - "apply_agent"
   - "interview_agent"
3. "parameters": any extracted parameters (job titles, locations, companies, etc.)
4. "explanation": brief explanation of your plan

Intent mapping:
- "Find me jobs and apply to them" → full_pipeline → all 4 agents
- "Search for Python engineer jobs" → job_search_only → [job_search_agent]
- "Tailor my CV for this job" → cv_only → [resume_agent]
- "Apply to the jobs I shortlisted" → apply_only → [resume_agent, apply_agent]
- "Prep me for interviews at Stripe" → interview_only → [interview_agent]

Important routing rule:
- If the user asks to find/search jobs and does NOT explicitly say "only search" or "just search",
  prefer full_pipeline so the flow continues with CV tailoring and application drafting.

Return ONLY valid JSON. No explanation outside JSON."""

INTENT_EXAMPLES = """
Examples:
Query: "Find senior ML engineer jobs in Pakistan and apply to the top 10"
Response: {"intent": "full_pipeline", "agents": ["job_search_agent", "resume_agent", "apply_agent"], "parameters": {"role": "Senior ML Engineer", "location": "Pakistan", "max_apply": 10}, "explanation": "Full pipeline: search → tailor CVs → apply"}

Query: "Just find me Python backend jobs remote"
Response: {"intent": "job_search_only", "agents": ["job_search_agent"], "parameters": {"role": "Python Backend Engineer", "location": "Remote"}, "explanation": "Search only, no applying"}

Query: "Help me prepare for my Google interview next week"
Response: {"intent": "interview_only", "agents": ["interview_agent"], "parameters": {"company": "Google"}, "explanation": "Interview prep only"}
"""


class OrchestratorAgent:
    def __init__(self, event_callback=None):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.event_callback = event_callback  # async fn to push SSE events to frontend
        self._graph: CompiledStateGraph | None = None
        self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine."""
        graph = StateGraph(dict)

        # Nodes
        graph.add_node("parse_intent", self._parse_intent_node)
        graph.add_node("route_agents", self._route_agents_node)
        graph.add_node("run_job_search", self._run_job_search_node)
        graph.add_node("run_resume", self._run_resume_node)
        graph.add_node("run_apply", self._run_apply_node)
        graph.add_node("run_interview", self._run_interview_node)
        graph.add_node("hitl_checkpoint", self._hitl_checkpoint_node)
        graph.add_node("finalize", self._finalize_node)

        # Entry
        graph.set_entry_point("parse_intent")

        # Edges
        graph.add_edge("parse_intent", "route_agents")
        graph.add_conditional_edges("route_agents", self._next_agent_router)
        graph.add_edge("run_job_search", "route_agents")
        graph.add_edge("run_resume", "hitl_checkpoint")   # CV always needs HITL
        graph.add_edge("hitl_checkpoint", "route_agents")
        graph.add_edge("run_apply", "finalize")
        graph.add_edge("run_interview", "finalize")
        graph.add_edge("finalize", END)

        self._graph = graph.compile()

    def _next_agent_router(self, state: dict) -> str:
        """Decide which agent node to run next."""
        agents_to_run = state.get("agents_to_run", [])
        completed = state.get("completed_agents", [])
        
        pending = [a for a in agents_to_run if a not in completed]
        if not pending:
            return "finalize"
        
        next_agent = pending[0]
        routing = {
            "job_search_agent": "run_job_search",
            "resume_agent": "run_resume",
            "apply_agent": "run_apply",
            "interview_agent": "run_interview",
        }
        return routing.get(next_agent, "finalize")

    async def _emit(self, event_type: str, message: str, agent_name: str = "", data: dict = {}):
        """Push a real-time event to the frontend via SSE."""
        event = AgentEvent(
            event_type=event_type,
            agent_name=agent_name or "orchestrator",
            message=message,
            data=data,
        )
        log.info("orchestrator.event", event_type=event_type, message=message)
        if self.event_callback:
            await self.event_callback(event)
        # Also cache in Redis for polling
        wf_id = data.get("workflow_id", "")
        if wf_id:
            await short_term.append_to_list(f"events:{wf_id}", event.model_dump(mode="json"))

    async def _parse_intent_node(self, state: dict) -> dict:
        """Use Claude to parse the user's query into structured intent."""
        await self._emit("agent_started", "Analyzing your request...", "orchestrator", 
                        {"workflow_id": state["workflow_id"]})

        import json
        parsed = None

        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT + "\n\n" + INTENT_EXAMPLES),
            HumanMessage(content=state["raw_query"])
        ]

        try:
            response = await self.llm.ainvoke(messages)
            parsed = json.loads(response.content)
        except Exception:
            q = state["raw_query"].lower()
            if "interview" in q or "prep" in q:
                parsed = {
                    "intent": "interview_only",
                    "agents": ["interview_agent"],
                    "parameters": {},
                    "explanation": "Using rule-based interview intent"
                }
            elif "cv" in q or "resume" in q or "tailor" in q:
                parsed = {
                    "intent": "cv_only",
                    "agents": ["resume_agent"],
                    "parameters": {},
                    "explanation": "Using rule-based CV intent"
                }
            elif "apply" in q:
                parsed = {
                    "intent": "apply_only",
                    "agents": ["job_search_agent", "resume_agent", "apply_agent"],
                    "parameters": {},
                    "explanation": "Using rule-based apply intent"
                }
            else:
                parsed = {
                    "intent": "job_search_only",
                    "agents": ["job_search_agent"],
                    "parameters": {},
                    "explanation": "Running job search only"
                }

        q = state["raw_query"].lower()
        asks_for_jobs = any(token in q for token in ("find", "search")) and "job" in q
        explicit_search_only = ("only search" in q) or ("just search" in q) or ("search only" in q)
        if asks_for_jobs and not explicit_search_only and parsed.get("intent") == "job_search_only":
            parsed["intent"] = "full_pipeline"
            parsed["agents"] = ["job_search_agent", "resume_agent", "apply_agent"]
            parsed["explanation"] = "Job search requested, continuing through CV and applications."
        
        await self._emit("agent_progress", 
                        f"Plan: {parsed.get('explanation', '')}",
                        "orchestrator",
                        {"workflow_id": state["workflow_id"], "plan": parsed})
        
        return {
            **state,
            "intent": parsed.get("intent", "full_pipeline"),
            "agents_to_run": parsed.get("agents", []),
            "agent_results": {"intent_params": parsed.get("parameters", {})},
        }

    async def _route_agents_node(self, state: dict) -> dict:
        """Routing node — just updates workflow status in Redis."""
        completed = state.get("completed_agents", [])
        pending = [a for a in state.get("agents_to_run", []) if a not in completed]
        
        await short_term.set_workflow_status(state["workflow_id"], {
            "status": AgentStatus.WORKING,
            "completed": completed,
            "pending": pending,
            "current": pending[0] if pending else None,
        })
        return state

    @trace_agent("job_search_agent")
    async def _run_job_search_node(self, state: dict) -> dict:
        """Run the Job Search Agent."""
        from agents.job_search_agent import JobSearchAgent
        
        await self._emit("agent_started", "Searching for jobs...", "job_search_agent",
                        {"workflow_id": state["workflow_id"]})
        
        agent = JobSearchAgent(event_callback=self.event_callback)
        params = state.get("agent_results", {}).get("intent_params", {})
        
        result = await agent.run(
            user_id=state["user_id"],
            workflow_id=state["workflow_id"],
            role=params.get("role", "Software Engineer"),
            locations=params.get("location", ["Remote"]) if isinstance(params.get("location"), list)
                      else [params.get("location", "Remote")],
            max_results=params.get("max_results", 20),
        )
        
        completed = state.get("completed_agents", []) + ["job_search_agent"]
        await self._emit("agent_done", f"Found {result.get('total', 0)} jobs", "job_search_agent",
                        {"workflow_id": state["workflow_id"], "total": result.get("total", 0)})
        
        return {**state, "completed_agents": completed, "agent_results": {**state.get("agent_results", {}), "job_search": result}}

    @trace_agent("resume_agent")
    async def _run_resume_node(self, state: dict) -> dict:
        """Run the Resume Agent."""
        from agents.resume_agent import ResumeAgent
        
        await self._emit("agent_started", "Tailoring CVs for each role...", "resume_agent",
                        {"workflow_id": state["workflow_id"]})
        
        agent = ResumeAgent(event_callback=self.event_callback)
        result = await agent.run(
            user_id=state["user_id"],
            workflow_id=state["workflow_id"],
        )
        
        completed = state.get("completed_agents", []) + ["resume_agent"]
        await self._emit("agent_done", f"{result.get('total', 0)} CVs ready for your review",
                        "resume_agent", {"workflow_id": state["workflow_id"]})
        
        return {**state, "completed_agents": completed, "hitl_required": True,
                "agent_results": {**state.get("agent_results", {}), "resume": result}}

    async def _hitl_checkpoint_node(self, state: dict) -> dict:
        """Pause and wait for user approval."""
        checkpoint_id = str(uuid.uuid4())
        auto_approve = os.getenv("AUTO_APPROVE_HITL", "false").lower() in ("1", "true", "yes")
        
        await short_term.set_hitl_checkpoint(checkpoint_id, {
            "type": "cv_review",
            "workflow_id": state["workflow_id"],
            "user_id": state["user_id"],
            "created_at": datetime.utcnow().isoformat(),
            "approved": None,
        })
        
        await short_term.set_workflow_status(state["workflow_id"], {
            "status": AgentStatus.AWAITING_APPROVAL,
            "hitl_checkpoint_id": checkpoint_id,
        })
        
        await self._emit(
            "hitl_required",
            "Please review and approve the tailored CVs before I continue.",
            "orchestrator",
            {"workflow_id": state["workflow_id"], "checkpoint_id": checkpoint_id, "type": "cv_review"}
        )

        if auto_approve:
            from schemas.models import CVStatus
            cvs = await long_term.get_cvs(state["user_id"], status=CVStatus.PENDING_APPROVAL)
            for cv in cvs:
                await long_term.update_cv_status(cv["id"], CVStatus.APPROVED)
            await short_term.resolve_hitl_checkpoint(checkpoint_id, True)
            await self._emit(
                "agent_progress",
                "Auto-approved CV checkpoint. Continuing workflow.",
                "orchestrator",
                {"workflow_id": state["workflow_id"]},
            )
            return {**state, "hitl_required": False, "hitl_checkpoint_id": checkpoint_id}
        
        # Wait for approval (polling Redis)
        import asyncio
        for _ in range(720):  # max 1 hour
            await asyncio.sleep(5)
            checkpoint = await short_term.get_hitl_checkpoint(checkpoint_id)
            if checkpoint and checkpoint.get("approved") is not None:
                if checkpoint["approved"]:
                    await self._emit("agent_progress", "CVs approved. Continuing...", "orchestrator",
                                    {"workflow_id": state["workflow_id"]})
                    break
                else:
                    await self._emit("agent_progress", "CVs rejected. Regenerating...", "orchestrator",
                                    {"workflow_id": state["workflow_id"]})
                    completed = [a for a in state.get("completed_agents", []) if a != "resume_agent"]
                    return {**state, "completed_agents": completed, "hitl_required": False}
        
        return {**state, "hitl_required": False, "hitl_checkpoint_id": checkpoint_id}

    @trace_agent("apply_agent")
    async def _run_apply_node(self, state: dict) -> dict:
        """Run the Apply Agent."""
        from agents.apply_agent import ApplyAgent
        
        await self._emit("agent_started", "Preparing application emails...", "apply_agent",
                        {"workflow_id": state["workflow_id"]})
        
        agent = ApplyAgent(event_callback=self.event_callback)
        result = await agent.run(
            user_id=state["user_id"],
            workflow_id=state["workflow_id"],
        )
        
        completed = state.get("completed_agents", []) + ["apply_agent"]
        await self._emit("agent_done", f"{result.get('sent', 0)} applications sent successfully!",
                        "apply_agent", {"workflow_id": state["workflow_id"]})
        
        return {**state, "completed_agents": completed,
                "agent_results": {**state.get("agent_results", {}), "apply": result}}

    @trace_agent("interview_agent")
    async def _run_interview_node(self, state: dict) -> dict:
        """Run the Interview Prep Agent."""
        from agents.interview_agent import InterviewAgent
        
        await self._emit("agent_started", "Generating interview preparation materials...", "interview_agent",
                        {"workflow_id": state["workflow_id"]})
        
        agent = InterviewAgent(event_callback=self.event_callback)
        result = await agent.run(
            user_id=state["user_id"],
            workflow_id=state["workflow_id"],
        )
        
        completed = state.get("completed_agents", []) + ["interview_agent"]
        await self._emit("agent_done", "Interview prep ready!", "interview_agent",
                        {"workflow_id": state["workflow_id"]})
        
        return {**state, "completed_agents": completed,
                "agent_results": {**state.get("agent_results", {}), "interview": result}}

    async def _finalize_node(self, state: dict) -> dict:
        """Finalize the workflow."""
        await short_term.set_workflow_status(state["workflow_id"], {
            "status": AgentStatus.COMPLETED,
            "completed_agents": state.get("completed_agents", []),
        })
        
        summary = f"All done. Completed: {', '.join(state.get('completed_agents', []))}"
        await self._emit("agent_done", summary, "orchestrator",
                        {"workflow_id": state["workflow_id"], "completed": True})
        
        return {**state, "status": AgentStatus.COMPLETED}

    async def run(
        self,
        user_id: str,
        session_id: str,
        raw_query: str,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Main entry point. Runs the full orchestrated workflow.
        Yields AgentEvent objects for SSE streaming.
        """
        workflow_id = str(uuid.uuid4())
        
        initial_state = {
            "workflow_id": workflow_id,
            "user_id": user_id,
            "session_id": session_id,
            "raw_query": raw_query,
            "intent": "",
            "agents_to_run": [],
            "completed_agents": [],
            "current_agent": "",
            "agent_results": {},
            "hitl_required": False,
            "hitl_checkpoint_id": "",
            "error": "",
            "status": AgentStatus.WORKING,
        }
        
        await short_term.set_workflow_status(workflow_id, initial_state)
        await short_term.add_message(session_id, {
            "role": "user",
            "content": raw_query,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        log.info("orchestrator.run_started", workflow_id=workflow_id, user_id=user_id)
        
        try:
            await self._graph.ainvoke(initial_state)
        except Exception as e:
            log.error("orchestrator.error", error=str(e), workflow_id=workflow_id)
            await self._emit("error", f"Error: {str(e)}", "orchestrator",
                           {"workflow_id": workflow_id})
        
        return workflow_id
