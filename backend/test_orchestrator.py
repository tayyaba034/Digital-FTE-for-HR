import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import OrchestratorAgent

async def main():
    agent = OrchestratorAgent()
    print("Testing Orchestrator Intent Parsing...")
    
    # We will invoke the _graph.ainvoke directly with a sample query
    initial_state = {
        "workflow_id": "test-123",
        "user_id": "u-123",
        "session_id": "s-123",
        "raw_query": "I want to find a python job in Pakistan.",
        "intent": "",
        "agents_to_run": [],
        "completed_agents": [],
        "current_agent": "",
        "agent_results": {},
        "hitl_required": False,
        "hitl_checkpoint_id": "",
        "error": "",
        "status": "WORKING",
    }
    
    # Run just the parse_intent node
    result = await agent._parse_intent_node(initial_state)
    print("Parsed Intent:", result.get("intent"))
    print("Agents to Run:", result.get("agents_to_run"))
    print("Parameters:", result.get("agent_results", {}).get("intent_params"))

if __name__ == "__main__":
    asyncio.run(main())
