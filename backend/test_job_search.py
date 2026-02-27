import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

from agents.job_search_agent import JobSearchAgent

async def main():
    agent = JobSearchAgent()
    print("Testing Job Search Agent with Apify...")
    
    result = await agent.run(
        user_id="u-123",
        workflow_id="w-123",
        role="Python Engineer",
        locations=["Pakistan"],
        max_results=3
    )
    
    print("Jobs Found:", result.get("total"))
    if result.get("total", 0) > 0:
        for job in result.get("jobs", [])[:2]:
            print(f"- {job.get('title')} at {job.get('company_name')} ({job.get('location')})")

if __name__ == "__main__":
    asyncio.run(main())
